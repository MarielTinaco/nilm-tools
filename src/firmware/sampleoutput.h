// This file was @generated automatically

#define SAMPLE_OUTPUT { \
  0x50811000, 0x000000ff, 0x00000001, 0x0000000b, 0x50401000, 0xffffffff, 0x00000001, 0x00000000, \
  0x50409000, 0xffffffff, 0x00000001, 0x00000000, 0x50411000, 0xffffffff, 0x00000001, 0x00000001, \
  0x50419000, 0xffffffff, 0x00000001, 0x1c011402, 0x50801000, 0xffffffff, 0x00000001, 0x07797403, \
  0x50809000, 0xffffffff, 0x00000001, 0x7f7f1e2b, 0x00000000 \
}
// 0 0 0 0        = 50401000 0xffffffff
// 0 0 0 0        = 50409000 0xffffffff
// 1 0 0 0        = 50411000 0xffffffff
// 2 14 1 1c      = 50419000 0xffffffff
// ?              = 50421000 0xffffffff
// ?              = 50429000 0xffffffff
// 3 74 79 7      = 50801000 0xffffffff
// 2b 1e 7f 7f    = 50809000 0xffffffff
// b 0 0 0        = 50811000 0xff000000


// [DEV] Prediction: addr: 50401000, val: 0
// addr: 50409000, val: 0
// addr: 50411000, val: 1
// addr: 50419000, val: 1c011402
// L_DATA32:
  // 0 0 0 0
  // 0 0 0 0
  // 1 0 0 0
  // 2 14 1 1c
  // 3 74 79 7
  // 2b 1e 7f 7f
  // b 0 0 0