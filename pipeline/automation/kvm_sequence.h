#pragma once
#include <vector>
#include "rule_matcher.h"

// Prints the planned KVM action sequence. Nothing is sent to a machine
// here -- the sequence is printed and waits for a person to approve it
// before it would ever run for real.
void render_kvm_sequence(const std::vector<Action>& actions);
