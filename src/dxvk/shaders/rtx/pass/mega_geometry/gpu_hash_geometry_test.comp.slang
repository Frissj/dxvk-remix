/*
* GPU Hash Test - Minimal shader to verify descriptor binding works
*/

#include "rtx/utility/common.slangh"

layout(binding = 2)
RWStructuredBuffer<uint> HashOutput;

struct PushConstants {
  uint testValue;
  uint pad1;
  uint pad2;
  uint pad3;
};

layout(push_constant)
ConstantBuffer<PushConstants> cb;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID) {
  // Just write a fixed constant to verify binding works
  HashOutput[0] = 0xDEADBEEF;
  HashOutput[1] = cb.testValue;
}
