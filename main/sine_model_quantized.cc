#include "sine_model_quantized.hpp"

const unsigned char sine_model_quantized_tflite[] = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x14, 0x00, 0x20, 0x00,
  0x1c, 0x00, 0x18, 0x00, 0x14, 0x00, 0x10, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x04, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
  0x88, 0x00, 0x00, 0x00, 0xe0, 0x00, 0x00, 0x00, 0xac, 0x03, 0x00, 0x00,
  0xbc, 0x03, 0x00, 0x00, 0x58, 0x0b, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x02, 0xfc, 0xff, 0xff,
  0x0c, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00,
  0x0f, 0x00, 0x00, 0x00, 0x73, 0x65, 0x72, 0x76, 0x69, 0x6e, 0x67, 0x5f,
  0x64, 0x65, 0x66, 0x61, 0x75, 0x6c, 0x74, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x94, 0xff, 0xff, 0xff, 0x0b, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x64, 0x65, 0x6e, 0x73,
  0x65, 0x5f, 0x35, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0xd6, 0xfc, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00,
  0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x33, 0x5f, 0x69, 0x6e, 0x70, 0x75,
  0x74, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0xdc, 0xff, 0xff, 0xff, 0x0e, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00, 0x43, 0x4f, 0x4e, 0x56,
  0x45, 0x52, 0x53, 0x49, 0x4f, 0x4e, 0x5f, 0x4d, 0x45, 0x54, 0x41, 0x44,
  0x41, 0x54, 0x41, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x08, 0x00, 0x04, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x13, 0x00, 0x00, 0x00, 0x6d, 0x69, 0x6e, 0x5f, 0x72, 0x75, 0x6e, 0x74,
  0x69, 0x6d, 0x65, 0x5f, 0x76, 0x65, 0x72, 0x73, 0x69, 0x6f, 0x6e, 0x00,
  0x0f, 0x00, 0x00, 0x00, 0xc8, 0x02, 0x00, 0x00, 0xc0, 0x02, 0x00, 0x00,
  0xac, 0x02, 0x00, 0x00, 0x84, 0x02, 0x00, 0x00, 0x34, 0x02, 0x00, 0x00,
  0x24, 0x01, 0x00, 0x00, 0xd4, 0x00, 0x00, 0x00, 0xb4, 0x00, 0x00, 0x00,
  0xac, 0x00, 0x00, 0x00, 0xa4, 0x00, 0x00, 0x00, 0x9c, 0x00, 0x00, 0x00,
  0x94, 0x00, 0x00, 0x00, 0x8c, 0x00, 0x00, 0x00, 0x6c, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x8e, 0xfd, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00,
  0x58, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0e, 0x00,
  0x08, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x28, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x08, 0x00, 0x04, 0x00,
  0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0xeb, 0x03, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x10, 0x00, 0x0c, 0x00,
  0x08, 0x00, 0x04, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
  0x32, 0x2e, 0x31, 0x32, 0x2e, 0x30, 0x00, 0x00, 0xf2, 0xfd, 0xff, 0xff,
  0x04, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x31, 0x2e, 0x31, 0x34,
  0x2e, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x74, 0xf6, 0xff, 0xff, 0x78, 0xf6, 0xff, 0xff, 0x7c, 0xf6, 0xff, 0xff,
  0x80, 0xf6, 0xff, 0xff, 0x84, 0xf6, 0xff, 0xff, 0x22, 0xfe, 0xff, 0xff,
  0x04, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x44, 0x66, 0x81, 0xcc,
  0x41, 0x1a, 0x09, 0x38, 0xb7, 0xcf, 0xd9, 0xec, 0xec, 0x40, 0xaa, 0xc8,
  0x3e, 0xfe, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
  0x8e, 0xf1, 0xff, 0xff, 0x96, 0xe0, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x1d, 0x00, 0x00, 0x00, 0x8a, 0x09, 0x00, 0x00,
  0x17, 0x0e, 0x00, 0x00, 0x5c, 0xeb, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xd7, 0x1d, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xa2, 0xe3, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x8a, 0xfe, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x00, 0x00, 0x06, 0x1f, 0x07, 0xe8, 0x07, 0xed, 0x0b, 0x2c,
  0xf7, 0xe2, 0x0e, 0xe6, 0x0d, 0x08, 0xea, 0x02, 0x07, 0x1b, 0xf1, 0xf1,
  0x0a, 0xef, 0xe3, 0x14, 0x1d, 0x03, 0xf5, 0xd8, 0x1b, 0x11, 0xe8, 0xf5,
  0xf1, 0x16, 0xfb, 0x1a, 0x08, 0x00, 0x07, 0x11, 0xfc, 0x0e, 0xeb, 0xe8,
  0xe7, 0x1d, 0xe4, 0xe7, 0x03, 0xe0, 0xf1, 0x18, 0xfc, 0x19, 0xf0, 0xe9,
  0xf7, 0xf0, 0x19, 0x04, 0x19, 0x0b, 0xf3, 0x0c, 0x07, 0x16, 0x08, 0xe6,
  0xf1, 0x0d, 0x06, 0x00, 0xf7, 0x18, 0xef, 0x21, 0xee, 0xf9, 0x07, 0x1d,
  0x08, 0x22, 0x12, 0xfa, 0xb7, 0xd4, 0x0f, 0x09, 0xee, 0x00, 0xf3, 0x1f,
  0xef, 0xde, 0x01, 0xf4, 0xea, 0xfe, 0x12, 0xf4, 0x01, 0x00, 0x16, 0xe1,
  0xef, 0x02, 0xf1, 0x1f, 0x12, 0xe8, 0x13, 0xfd, 0x17, 0xfd, 0x08, 0x08,
  0xf5, 0x0e, 0xf9, 0xec, 0xeb, 0x15, 0xfe, 0xe9, 0x1c, 0x02, 0x17, 0x01,
  0xfe, 0x0e, 0xe9, 0xea, 0x26, 0xf9, 0x09, 0xf3, 0xf2, 0x09, 0x01, 0x04,
  0x15, 0xde, 0xe8, 0x1a, 0xe8, 0xf6, 0x08, 0xfa, 0xf9, 0x07, 0x08, 0xd1,
  0x02, 0x0c, 0x0e, 0x19, 0x14, 0xf6, 0xf4, 0x0a, 0xed, 0xce, 0xf2, 0xff,
  0x81, 0xba, 0x19, 0x10, 0x17, 0xf8, 0x17, 0x21, 0xef, 0x13, 0x03, 0xe7,
  0x06, 0xdc, 0x06, 0xfd, 0x00, 0x10, 0x13, 0xdf, 0xfd, 0xe5, 0xe4, 0x28,
  0xf5, 0xeb, 0x14, 0xef, 0x0c, 0xf5, 0xff, 0x1c, 0x01, 0x0b, 0x06, 0xfa,
  0xfe, 0x0b, 0x1c, 0xe6, 0x1d, 0xfb, 0xe5, 0xec, 0xf5, 0xee, 0xfe, 0x04,
  0xf3, 0xe9, 0xe2, 0xfe, 0x16, 0xea, 0xf2, 0x11, 0x06, 0x11, 0xef, 0x11,
  0xff, 0x02, 0xe3, 0x12, 0x0d, 0x04, 0xed, 0x00, 0x1a, 0x14, 0xf6, 0x08,
  0xea, 0x08, 0x18, 0xe5, 0xb6, 0xb6, 0xec, 0x0b, 0x11, 0x18, 0x18, 0xf7,
  0xf4, 0xf1, 0xf8, 0xe5, 0xf9, 0xff, 0x06, 0xfa, 0x96, 0xff, 0xff, 0xff,
  0x04, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x8a, 0xe9, 0xff, 0xff,
  0x7b, 0xe7, 0xff, 0xff, 0x81, 0xe5, 0xff, 0xff, 0x30, 0xfc, 0xff, 0xff,
  0xfc, 0x0b, 0x00, 0x00, 0xba, 0x0b, 0x00, 0x00, 0x7e, 0x0e, 0x00, 0x00,
  0xfa, 0xfd, 0xff, 0xff, 0x94, 0x0c, 0x00, 0x00, 0x71, 0x0f, 0x00, 0x00,
  0xdf, 0x0e, 0x00, 0x00, 0x50, 0x0e, 0x00, 0x00, 0x53, 0xfe, 0xff, 0xff,
  0x66, 0xff, 0xff, 0xff, 0xde, 0xf0, 0xff, 0xff, 0x12, 0x00, 0x00, 0x00,
  0xe2, 0xff, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x59, 0x7f, 0x20, 0xeb, 0xe1, 0x88, 0x77, 0x01, 0xc5, 0x50, 0x96, 0x56,
  0x1b, 0x33, 0x01, 0xbb, 0x00, 0x00, 0x06, 0x00, 0x08, 0x00, 0x04, 0x00,
  0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0xe7, 0xc4, 0xff, 0xff, 0x7c, 0xf8, 0xff, 0xff, 0x80, 0xf8, 0xff, 0xff,
  0x0f, 0x00, 0x00, 0x00, 0x4d, 0x4c, 0x49, 0x52, 0x20, 0x43, 0x6f, 0x6e,
  0x76, 0x65, 0x72, 0x74, 0x65, 0x64, 0x2e, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x18, 0x00, 0x14, 0x00,
  0x10, 0x00, 0x0c, 0x00, 0x08, 0x00, 0x04, 0x00, 0x0e, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x40, 0x01, 0x00, 0x00,
  0x44, 0x01, 0x00, 0x00, 0x48, 0x01, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x6d, 0x61, 0x69, 0x6e, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
  0x08, 0x01, 0x00, 0x00, 0xb8, 0x00, 0x00, 0x00, 0x6c, 0x00, 0x00, 0x00,
  0x34, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x00,
  0x10, 0x00, 0x0c, 0x00, 0x08, 0x00, 0x04, 0x00, 0x0a, 0x00, 0x00, 0x00,
  0x0c, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x0a, 0x00, 0x00, 0x00, 0x92, 0xff, 0xff, 0xff, 0x14, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x08, 0x10, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x30, 0xf9, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00,
  0x0a, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xc6, 0xff, 0xff, 0xff,
  0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x14, 0x00, 0x00, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xb6, 0xff, 0xff, 0xff,
  0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x1a, 0x00, 0x14, 0x00,
  0x10, 0x00, 0x0c, 0x00, 0x0b, 0x00, 0x04, 0x00, 0x0e, 0x00, 0x00, 0x00,
  0x1c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x1c, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00,
  0x08, 0x00, 0x07, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
  0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x07, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x0a, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x08, 0x00, 0x04, 0x00,
  0x0a, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0xc8, 0x05, 0x00, 0x00, 0x28, 0x05, 0x00, 0x00, 0xa0, 0x04, 0x00, 0x00,
  0x24, 0x04, 0x00, 0x00, 0xb8, 0x03, 0x00, 0x00, 0x3c, 0x03, 0x00, 0x00,
  0xd0, 0x02, 0x00, 0x00, 0x5c, 0x02, 0x00, 0x00, 0x90, 0x01, 0x00, 0x00,
  0xdc, 0x00, 0x00, 0x00, 0x60, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x7e, 0xfa, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01, 0x14, 0x00, 0x00, 0x00,
  0x1c, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x34, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
  0x01, 0x00, 0x00, 0x00, 0x68, 0xfa, 0xff, 0xff, 0x19, 0x00, 0x00, 0x00,
  0x53, 0x74, 0x61, 0x74, 0x65, 0x66, 0x75, 0x6c, 0x50, 0x61, 0x72, 0x74,
  0x69, 0x74, 0x69, 0x6f, 0x6e, 0x65, 0x64, 0x43, 0x61, 0x6c, 0x6c, 0x3a,
  0x30, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x26, 0xfe, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
  0x18, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00,
  0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x50, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00,
  0x6c, 0xfb, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0xec, 0xc0, 0x03, 0x3c, 0x1a, 0x00, 0x00, 0x00,
  0x53, 0x74, 0x61, 0x74, 0x65, 0x66, 0x75, 0x6c, 0x50, 0x61, 0x72, 0x74,
  0x69, 0x74, 0x69, 0x6f, 0x6e, 0x65, 0x64, 0x43, 0x61, 0x6c, 0x6c, 0x3a,
  0x30, 0x31, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x9e, 0xfe, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
  0x18, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00,
  0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x88, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x10, 0x00, 0x00, 0x00,
  0xe4, 0xfb, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x80, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  0x01, 0x00, 0x00, 0x00, 0xa2, 0xd9, 0x84, 0x3b, 0x52, 0x00, 0x00, 0x00,
  0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x74, 0x69, 0x61, 0x6c, 0x5f, 0x31,
  0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x34, 0x2f, 0x4d, 0x61, 0x74,
  0x4d, 0x75, 0x6c, 0x3b, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x74, 0x69,
  0x61, 0x6c, 0x5f, 0x31, 0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x34,
  0x2f, 0x52, 0x65, 0x6c, 0x75, 0x3b, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e,
  0x74, 0x69, 0x61, 0x6c, 0x5f, 0x31, 0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65,
  0x5f, 0x34, 0x2f, 0x42, 0x69, 0x61, 0x73, 0x41, 0x64, 0x64, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x4e, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01, 0x18, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x09, 0x88, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0xff, 0xff, 0xff, 0xff, 0x10, 0x00, 0x00, 0x00, 0x94, 0xfc, 0xff, 0xff,
  0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x80, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00,
  0x9b, 0xf0, 0xf5, 0x3b, 0x52, 0x00, 0x00, 0x00, 0x73, 0x65, 0x71, 0x75,
  0x65, 0x6e, 0x74, 0x69, 0x61, 0x6c, 0x5f, 0x31, 0x2f, 0x64, 0x65, 0x6e,
  0x73, 0x65, 0x5f, 0x33, 0x2f, 0x4d, 0x61, 0x74, 0x4d, 0x75, 0x6c, 0x3b,
  0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x74, 0x69, 0x61, 0x6c, 0x5f, 0x31,
  0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x33, 0x2f, 0x52, 0x65, 0x6c,
  0x75, 0x3b, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x74, 0x69, 0x61, 0x6c,
  0x5f, 0x31, 0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x33, 0x2f, 0x42,
  0x69, 0x61, 0x73, 0x41, 0x64, 0x64, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x16, 0x00,
  0x20, 0x00, 0x1c, 0x00, 0x1b, 0x00, 0x14, 0x00, 0x10, 0x00, 0x0c, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x07, 0x00, 0x16, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x18, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
  0x40, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09,
  0x48, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
  0x01, 0x00, 0x00, 0x00, 0x5c, 0xfd, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x80, 0xff, 0xff, 0xff,
  0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x9a, 0xcb, 0xc9, 0x3c, 0x0c, 0x00, 0x00, 0x00, 0x74, 0x66, 0x6c, 0x2e,
  0x71, 0x75, 0x61, 0x6e, 0x74, 0x69, 0x7a, 0x65, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0xd2, 0xfd, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01, 0x14, 0x00, 0x00, 0x00,
  0x30, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09,
  0x44, 0x00, 0x00, 0x00, 0xbc, 0xfd, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x4e, 0xd9, 0x8d, 0x3b,
  0x1b, 0x00, 0x00, 0x00, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x74, 0x69,
  0x61, 0x6c, 0x5f, 0x31, 0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x33,
  0x2f, 0x4d, 0x61, 0x74, 0x4d, 0x75, 0x6c, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x3a, 0xfe, 0xff, 0xff,
  0x00, 0x00, 0x00, 0x01, 0x14, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00,
  0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x58, 0x00, 0x00, 0x00,
  0x24, 0xfe, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xde, 0xa0, 0xdf, 0x38,
  0x2b, 0x00, 0x00, 0x00, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x74, 0x69,
  0x61, 0x6c, 0x5f, 0x31, 0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x33,
  0x2f, 0x42, 0x69, 0x61, 0x73, 0x41, 0x64, 0x64, 0x2f, 0x52, 0x65, 0x61,
  0x64, 0x56, 0x61, 0x72, 0x69, 0x61, 0x62, 0x6c, 0x65, 0x4f, 0x70, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0xb2, 0xfe, 0xff, 0xff,
  0x00, 0x00, 0x00, 0x01, 0x14, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00,
  0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x44, 0x00, 0x00, 0x00,
  0x9c, 0xfe, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0xa1, 0x28, 0x6d, 0x3c, 0x1b, 0x00, 0x00, 0x00,
  0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x74, 0x69, 0x61, 0x6c, 0x5f, 0x31,
  0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x34, 0x2f, 0x4d, 0x61, 0x74,
  0x4d, 0x75, 0x6c, 0x00, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x1a, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
  0x14, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x02, 0x58, 0x00, 0x00, 0x00, 0x04, 0xff, 0xff, 0xff,
  0x08, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0xc8, 0xd6, 0xe3, 0x38, 0x2b, 0x00, 0x00, 0x00,
  0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x74, 0x69, 0x61, 0x6c, 0x5f, 0x31,
  0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x34, 0x2f, 0x42, 0x69, 0x61,
  0x73, 0x41, 0x64, 0x64, 0x2f, 0x52, 0x65, 0x61, 0x64, 0x56, 0x61, 0x72,
  0x69, 0x61, 0x62, 0x6c, 0x65, 0x4f, 0x70, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x92, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
  0x14, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x09, 0x48, 0x00, 0x00, 0x00, 0x7c, 0xff, 0xff, 0xff,
  0x08, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x5d, 0xcc, 0x0d, 0x3c, 0x1b, 0x00, 0x00, 0x00,
  0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x74, 0x69, 0x61, 0x6c, 0x5f, 0x31,
  0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x35, 0x2f, 0x4d, 0x61, 0x74,
  0x4d, 0x75, 0x6c, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x16, 0x00, 0x1c, 0x00, 0x18, 0x00,
  0x17, 0x00, 0x10, 0x00, 0x0c, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x07, 0x00, 0x16, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
  0x20, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x02, 0x64, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x0c, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x04, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0xd8, 0x2b, 0x13, 0x38, 0x2b, 0x00, 0x00, 0x00,
  0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x74, 0x69, 0x61, 0x6c, 0x5f, 0x31,
  0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x35, 0x2f, 0x42, 0x69, 0x61,
  0x73, 0x41, 0x64, 0x64, 0x2f, 0x52, 0x65, 0x61, 0x64, 0x56, 0x61, 0x72,
  0x69, 0x61, 0x62, 0x6c, 0x65, 0x4f, 0x70, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x16, 0x00, 0x1c, 0x00, 0x18, 0x00,
  0x00, 0x00, 0x14, 0x00, 0x10, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x07, 0x00, 0x16, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
  0x14, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0xff, 0xff, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x04, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00, 0x73, 0x65, 0x72, 0x76,
  0x69, 0x6e, 0x67, 0x5f, 0x64, 0x65, 0x66, 0x61, 0x75, 0x6c, 0x74, 0x5f,
  0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x33, 0x5f, 0x69, 0x6e, 0x70, 0x75,
  0x74, 0x3a, 0x30, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x44, 0x00, 0x00, 0x00,
  0x24, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xf0, 0xff, 0xff, 0xff,
  0x06, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06,
  0x0c, 0x00, 0x10, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x08, 0x00, 0x04, 0x00,
  0x0c, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x09, 0x0c, 0x00, 0x0c, 0x00, 0x0b, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x04, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x72, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x72
};
const unsigned int sine_model_quantized_tflite_len = 3040;
