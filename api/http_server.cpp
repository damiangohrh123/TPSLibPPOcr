#include "api/http_server.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

// Case-insensitive lookup of the Content-Length header's value within the
// raw header block. Returns -1 if the header isn't present.
long find_content_length(const std::string& headers) {
    std::string lower = headers;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                    [](unsigned char c) { return std::tolower(c); });
    size_t pos = lower.find("content-length:");
    if (pos == std::string::npos) return -1;
    pos += std::strlen("content-length:");
    while (pos < headers.size() && headers[pos] == ' ') ++pos;
    return std::strtol(headers.c_str() + pos, nullptr, 10);
}

const char* reason_phrase(int status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 500: return "Internal Server Error";
        default:  return "Unknown";
    }
}

}  // namespace

HttpServer::HttpServer(int port) : port_(port) {}

void HttpServer::on(const std::string& method, const std::string& path, Handler handler) {
    routes_.emplace_back(method, path, std::move(handler));
}

void HttpServer::run() {
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        perror("socket");
        return;
    }
    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port_));

    if (bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        perror("bind");
        close(listen_fd);
        return;
    }
    if (listen(listen_fd, /*backlog=*/16) < 0) {
        perror("listen");
        close(listen_fd);
        return;
    }

    for (;;) {
        int client_fd = accept(listen_fd, nullptr, nullptr);
        if (client_fd < 0) {
            perror("accept");
            continue;
        }

        // Read until the full header block ("\r\n\r\n") has arrived.
        std::string buf;
        char chunk[4096];
        size_t header_end = std::string::npos;
        while (header_end == std::string::npos) {
            ssize_t n = recv(client_fd, chunk, sizeof(chunk), 0);
            if (n <= 0) break;
            buf.append(chunk, static_cast<size_t>(n));
            header_end = buf.find("\r\n\r\n");
        }
        if (header_end == std::string::npos) {
            close(client_fd);
            continue;
        }

        std::string header_block = buf.substr(0, header_end);
        std::string body = buf.substr(header_end + 4);

        // Parse the request line: "METHOD PATH HTTP/1.1".
        size_t line_end = header_block.find("\r\n");
        std::string request_line = header_block.substr(0, line_end);
        size_t sp1 = request_line.find(' ');
        size_t sp2 = request_line.find(' ', sp1 + 1);

        HttpRequest req;
        HttpResponse res;
        if (sp1 == std::string::npos || sp2 == std::string::npos) {
            res.status = 400;
            res.body = "{\"error\":\"malformed request line\"}";
        } else {
            req.method = request_line.substr(0, sp1);
            req.path = request_line.substr(sp1 + 1, sp2 - sp1 - 1);

            // Keep reading until the declared Content-Length has fully
            // arrived -- a single recv() is not guaranteed to return the
            // whole body in one call.
            long content_length = find_content_length(header_block);
            if (content_length > 0) {
                while (static_cast<long>(body.size()) < content_length) {
                    ssize_t n = recv(client_fd, chunk, sizeof(chunk), 0);
                    if (n <= 0) break;
                    body.append(chunk, static_cast<size_t>(n));
                }
            }
            req.body = std::move(body);

            res.status = 404;
            res.body = "{\"error\":\"not found\"}";
            for (const auto& [method, path, handler] : routes_) {
                if (method == req.method && path == req.path) {
                    res = handler(req);
                    break;
                }
            }
        }

        std::ostringstream out;
        out << "HTTP/1.1 " << res.status << " " << reason_phrase(res.status) << "\r\n"
            << "Content-Type: " << res.content_type << "\r\n"
            << "Content-Length: " << res.body.size() << "\r\n"
            << "Connection: close\r\n\r\n"
            << res.body;
        std::string response_str = out.str();
        send(client_fd, response_str.data(), response_str.size(), 0);
        close(client_fd);
    }
}
