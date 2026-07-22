#include "kvm_sequence.h"
#include <cstdio>

void render_kvm_sequence(const std::vector<Action>& actions) {
    printf("Planned KVM action sequence (not yet sent to the machine):\n\n");
    for (const auto& a : actions) {
        if (a.type == "move") {
            printf("  MOVE   (%d, %d)\n", a.x, a.y);
        } else if (a.type == "click") {
            printf("  CLICK\n");
        } else if (a.type == "type") {
            for (char ch : a.text) {
                printf("  KEYDOWN \"%c\" / KEYUP \"%c\"\n", ch, ch);
            }
            printf("  KEYDOWN \"Enter\" / KEYUP \"Enter\"\n");
        }
    }
    printf("\nWaiting for a person to approve this sequence before it runs.\n");
}
