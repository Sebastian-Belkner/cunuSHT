# 1 "CMakeCUDACompilerId.cu"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
# 1
#pragma GCC diagnostic push
# 1
#pragma GCC diagnostic ignored "-Wunused-variable"
# 1
#pragma GCC diagnostic ignored "-Wunused-function"
# 1
static char __nv_inited_managed_rt = 0; static void **__nv_fatbinhandle_for_managed_rt; static void __nv_save_fatbinhandle_for_managed_rt(void **in){__nv_fatbinhandle_for_managed_rt = in;} static char __nv_init_managed_rt_with_module(void **); static inline void __nv_init_managed_rt(void) { __nv_inited_managed_rt = (__nv_inited_managed_rt ? __nv_inited_managed_rt                 : __nv_init_managed_rt_with_module(__nv_fatbinhandle_for_managed_rt));}
# 1
#pragma GCC diagnostic pop
# 1
#pragma GCC diagnostic ignored "-Wunused-variable"

# 1
#define __nv_is_extended_device_lambda_closure_type(X) false
#define __nv_is_extended_host_device_lambda_closure_type(X) false
#if defined(__nv_is_extended_device_lambda_closure_type) && defined(__nv_is_extended_host_device_lambda_closure_type)
#endif

# 1
# 61 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
#pragma GCC diagnostic push
# 64
#pragma GCC diagnostic ignored "-Wunused-function"
# 68 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_types.h" 3
#if 0
# 68
enum cudaRoundMode { 
# 70
cudaRoundNearest, 
# 71
cudaRoundZero, 
# 72
cudaRoundPosInf, 
# 73
cudaRoundMinInf
# 74
}; 
#endif
# 100 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 100
struct char1 { 
# 102
signed char x; 
# 103
}; 
#endif
# 105 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 105
struct uchar1 { 
# 107
unsigned char x; 
# 108
}; 
#endif
# 111 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 111
struct __attribute((aligned(2))) char2 { 
# 113
signed char x, y; 
# 114
}; 
#endif
# 116 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 116
struct __attribute((aligned(2))) uchar2 { 
# 118
unsigned char x, y; 
# 119
}; 
#endif
# 121 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 121
struct char3 { 
# 123
signed char x, y, z; 
# 124
}; 
#endif
# 126 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 126
struct uchar3 { 
# 128
unsigned char x, y, z; 
# 129
}; 
#endif
# 131 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 131
struct __attribute((aligned(4))) char4 { 
# 133
signed char x, y, z, w; 
# 134
}; 
#endif
# 136 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 136
struct __attribute((aligned(4))) uchar4 { 
# 138
unsigned char x, y, z, w; 
# 139
}; 
#endif
# 141 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 141
struct short1 { 
# 143
short x; 
# 144
}; 
#endif
# 146 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 146
struct ushort1 { 
# 148
unsigned short x; 
# 149
}; 
#endif
# 151 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 151
struct __attribute((aligned(4))) short2 { 
# 153
short x, y; 
# 154
}; 
#endif
# 156 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 156
struct __attribute((aligned(4))) ushort2 { 
# 158
unsigned short x, y; 
# 159
}; 
#endif
# 161 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 161
struct short3 { 
# 163
short x, y, z; 
# 164
}; 
#endif
# 166 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 166
struct ushort3 { 
# 168
unsigned short x, y, z; 
# 169
}; 
#endif
# 171 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 171
struct __attribute((aligned(8))) short4 { short x; short y; short z; short w; }; 
#endif
# 172 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 172
struct __attribute((aligned(8))) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; }; 
#endif
# 174 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 174
struct int1 { 
# 176
int x; 
# 177
}; 
#endif
# 179 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 179
struct uint1 { 
# 181
unsigned x; 
# 182
}; 
#endif
# 184 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 184
struct __attribute((aligned(8))) int2 { int x; int y; }; 
#endif
# 185 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 185
struct __attribute((aligned(8))) uint2 { unsigned x; unsigned y; }; 
#endif
# 187 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 187
struct int3 { 
# 189
int x, y, z; 
# 190
}; 
#endif
# 192 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 192
struct uint3 { 
# 194
unsigned x, y, z; 
# 195
}; 
#endif
# 197 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 197
struct __attribute((aligned(16))) int4 { 
# 199
int x, y, z, w; 
# 200
}; 
#endif
# 202 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 202
struct __attribute((aligned(16))) uint4 { 
# 204
unsigned x, y, z, w; 
# 205
}; 
#endif
# 207 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 207
struct long1 { 
# 209
long x; 
# 210
}; 
#endif
# 212 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 212
struct ulong1 { 
# 214
unsigned long x; 
# 215
}; 
#endif
# 222 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 222
struct __attribute((aligned((2) * sizeof(long)))) long2 { 
# 224
long x, y; 
# 225
}; 
#endif
# 227 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 227
struct __attribute((aligned((2) * sizeof(unsigned long)))) ulong2 { 
# 229
unsigned long x, y; 
# 230
}; 
#endif
# 234 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 234
struct long3 { 
# 236
long x, y, z; 
# 237
}; 
#endif
# 239 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 239
struct ulong3 { 
# 241
unsigned long x, y, z; 
# 242
}; 
#endif
# 244 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 244
struct __attribute((aligned(16))) long4 { 
# 246
long x, y, z, w; 
# 247
}; 
#endif
# 249 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 249
struct __attribute((aligned(16))) ulong4 { 
# 251
unsigned long x, y, z, w; 
# 252
}; 
#endif
# 254 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 254
struct float1 { 
# 256
float x; 
# 257
}; 
#endif
# 276 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 276
struct __attribute((aligned(8))) float2 { float x; float y; }; 
#endif
# 281 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 281
struct float3 { 
# 283
float x, y, z; 
# 284
}; 
#endif
# 286 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 286
struct __attribute((aligned(16))) float4 { 
# 288
float x, y, z, w; 
# 289
}; 
#endif
# 291 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 291
struct longlong1 { 
# 293
long long x; 
# 294
}; 
#endif
# 296 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 296
struct ulonglong1 { 
# 298
unsigned long long x; 
# 299
}; 
#endif
# 301 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 301
struct __attribute((aligned(16))) longlong2 { 
# 303
long long x, y; 
# 304
}; 
#endif
# 306 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 306
struct __attribute((aligned(16))) ulonglong2 { 
# 308
unsigned long long x, y; 
# 309
}; 
#endif
# 311 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 311
struct longlong3 { 
# 313
long long x, y, z; 
# 314
}; 
#endif
# 316 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 316
struct ulonglong3 { 
# 318
unsigned long long x, y, z; 
# 319
}; 
#endif
# 321 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 321
struct __attribute((aligned(16))) longlong4 { 
# 323
long long x, y, z, w; 
# 324
}; 
#endif
# 326 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 326
struct __attribute((aligned(16))) ulonglong4 { 
# 328
unsigned long long x, y, z, w; 
# 329
}; 
#endif
# 331 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 331
struct double1 { 
# 333
double x; 
# 334
}; 
#endif
# 336 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 336
struct __attribute((aligned(16))) double2 { 
# 338
double x, y; 
# 339
}; 
#endif
# 341 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 341
struct double3 { 
# 343
double x, y, z; 
# 344
}; 
#endif
# 346 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 346
struct __attribute((aligned(16))) double4 { 
# 348
double x, y, z, w; 
# 349
}; 
#endif
# 363 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef char1 
# 363
char1; 
#endif
# 364 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef uchar1 
# 364
uchar1; 
#endif
# 365 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef char2 
# 365
char2; 
#endif
# 366 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef uchar2 
# 366
uchar2; 
#endif
# 367 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef char3 
# 367
char3; 
#endif
# 368 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef uchar3 
# 368
uchar3; 
#endif
# 369 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef char4 
# 369
char4; 
#endif
# 370 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef uchar4 
# 370
uchar4; 
#endif
# 371 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef short1 
# 371
short1; 
#endif
# 372 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef ushort1 
# 372
ushort1; 
#endif
# 373 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef short2 
# 373
short2; 
#endif
# 374 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef ushort2 
# 374
ushort2; 
#endif
# 375 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef short3 
# 375
short3; 
#endif
# 376 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef ushort3 
# 376
ushort3; 
#endif
# 377 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef short4 
# 377
short4; 
#endif
# 378 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef ushort4 
# 378
ushort4; 
#endif
# 379 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef int1 
# 379
int1; 
#endif
# 380 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef uint1 
# 380
uint1; 
#endif
# 381 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef int2 
# 381
int2; 
#endif
# 382 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef uint2 
# 382
uint2; 
#endif
# 383 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef int3 
# 383
int3; 
#endif
# 384 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef uint3 
# 384
uint3; 
#endif
# 385 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef int4 
# 385
int4; 
#endif
# 386 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef uint4 
# 386
uint4; 
#endif
# 387 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef long1 
# 387
long1; 
#endif
# 388 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef ulong1 
# 388
ulong1; 
#endif
# 389 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef long2 
# 389
long2; 
#endif
# 390 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef ulong2 
# 390
ulong2; 
#endif
# 391 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef long3 
# 391
long3; 
#endif
# 392 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef ulong3 
# 392
ulong3; 
#endif
# 393 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef long4 
# 393
long4; 
#endif
# 394 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef ulong4 
# 394
ulong4; 
#endif
# 395 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef float1 
# 395
float1; 
#endif
# 396 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef float2 
# 396
float2; 
#endif
# 397 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef float3 
# 397
float3; 
#endif
# 398 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef float4 
# 398
float4; 
#endif
# 399 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef longlong1 
# 399
longlong1; 
#endif
# 400 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef ulonglong1 
# 400
ulonglong1; 
#endif
# 401 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef longlong2 
# 401
longlong2; 
#endif
# 402 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef ulonglong2 
# 402
ulonglong2; 
#endif
# 403 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef longlong3 
# 403
longlong3; 
#endif
# 404 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef ulonglong3 
# 404
ulonglong3; 
#endif
# 405 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef longlong4 
# 405
longlong4; 
#endif
# 406 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef ulonglong4 
# 406
ulonglong4; 
#endif
# 407 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef double1 
# 407
double1; 
#endif
# 408 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef double2 
# 408
double2; 
#endif
# 409 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef double3 
# 409
double3; 
#endif
# 410 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef double4 
# 410
double4; 
#endif
# 418 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
# 418
struct dim3 { 
# 420
unsigned x, y, z; 
# 432
}; 
#endif
# 434 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_types.h" 3
#if 0
typedef dim3 
# 434
dim3; 
#endif
# 149 "/usr/lib/gcc/x86_64-redhat-linux/8/include/stddef.h" 3
typedef long ptrdiff_t; 
# 216 "/usr/lib/gcc/x86_64-redhat-linux/8/include/stddef.h" 3
typedef unsigned long size_t; 
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
# 437 "/usr/lib/gcc/x86_64-redhat-linux/8/include/stddef.h" 3
typedef 
# 426 "/usr/lib/gcc/x86_64-redhat-linux/8/include/stddef.h" 3
struct { 
# 427
long long __max_align_ll __attribute((__aligned__(__alignof__(long long)))); 
# 428
long double __max_align_ld __attribute((__aligned__(__alignof__(long double)))); 
# 437 "/usr/lib/gcc/x86_64-redhat-linux/8/include/stddef.h" 3
} max_align_t; 
# 444
typedef __decltype((nullptr)) nullptr_t; 
# 202 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 202
enum cudaError { 
# 209
cudaSuccess, 
# 215
cudaErrorInvalidValue, 
# 221
cudaErrorMemoryAllocation, 
# 227
cudaErrorInitializationError, 
# 234
cudaErrorCudartUnloading, 
# 241
cudaErrorProfilerDisabled, 
# 249
cudaErrorProfilerNotInitialized, 
# 256
cudaErrorProfilerAlreadyStarted, 
# 263
cudaErrorProfilerAlreadyStopped, 
# 272 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorInvalidConfiguration, 
# 278
cudaErrorInvalidPitchValue = 12, 
# 284
cudaErrorInvalidSymbol, 
# 292
cudaErrorInvalidHostPointer = 16, 
# 300
cudaErrorInvalidDevicePointer, 
# 306
cudaErrorInvalidTexture, 
# 312
cudaErrorInvalidTextureBinding, 
# 319
cudaErrorInvalidChannelDescriptor, 
# 325
cudaErrorInvalidMemcpyDirection, 
# 335 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorAddressOfConstant, 
# 344 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorTextureFetchFailed, 
# 353 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorTextureNotBound, 
# 362 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorSynchronizationError, 
# 368
cudaErrorInvalidFilterSetting, 
# 374
cudaErrorInvalidNormSetting, 
# 382
cudaErrorMixedDeviceExecution, 
# 390
cudaErrorNotYetImplemented = 31, 
# 399 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorMemoryValueTooLarge, 
# 406
cudaErrorStubLibrary = 34, 
# 413
cudaErrorInsufficientDriver, 
# 420
cudaErrorCallRequiresNewerDriver, 
# 426
cudaErrorInvalidSurface, 
# 432
cudaErrorDuplicateVariableName = 43, 
# 438
cudaErrorDuplicateTextureName, 
# 444
cudaErrorDuplicateSurfaceName, 
# 454 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorDevicesUnavailable, 
# 467 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorIncompatibleDriverContext = 49, 
# 473
cudaErrorMissingConfiguration = 52, 
# 482 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorPriorLaunchFailure, 
# 489
cudaErrorLaunchMaxDepthExceeded = 65, 
# 497
cudaErrorLaunchFileScopedTex, 
# 505
cudaErrorLaunchFileScopedSurf, 
# 520 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorSyncDepthExceeded, 
# 532 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorLaunchPendingCountExceeded, 
# 538
cudaErrorInvalidDeviceFunction = 98, 
# 544
cudaErrorNoDevice = 100, 
# 551
cudaErrorInvalidDevice, 
# 556
cudaErrorDeviceNotLicensed, 
# 565 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorSoftwareValidityNotEstablished, 
# 570
cudaErrorStartupFailure = 127, 
# 575
cudaErrorInvalidKernelImage = 200, 
# 585 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorDeviceUninitialized, 
# 590
cudaErrorMapBufferObjectFailed = 205, 
# 595
cudaErrorUnmapBufferObjectFailed, 
# 601
cudaErrorArrayIsMapped, 
# 606
cudaErrorAlreadyMapped, 
# 614
cudaErrorNoKernelImageForDevice, 
# 619
cudaErrorAlreadyAcquired, 
# 624
cudaErrorNotMapped, 
# 630
cudaErrorNotMappedAsArray, 
# 636
cudaErrorNotMappedAsPointer, 
# 642
cudaErrorECCUncorrectable, 
# 648
cudaErrorUnsupportedLimit, 
# 654
cudaErrorDeviceAlreadyInUse, 
# 660
cudaErrorPeerAccessUnsupported, 
# 666
cudaErrorInvalidPtx, 
# 671
cudaErrorInvalidGraphicsContext, 
# 677
cudaErrorNvlinkUncorrectable, 
# 684
cudaErrorJitCompilerNotFound, 
# 691
cudaErrorUnsupportedPtxVersion, 
# 698
cudaErrorJitCompilationDisabled, 
# 703
cudaErrorUnsupportedExecAffinity, 
# 708
cudaErrorInvalidSource = 300, 
# 713
cudaErrorFileNotFound, 
# 718
cudaErrorSharedObjectSymbolNotFound, 
# 723
cudaErrorSharedObjectInitFailed, 
# 728
cudaErrorOperatingSystem, 
# 735
cudaErrorInvalidResourceHandle = 400, 
# 741
cudaErrorIllegalState, 
# 748
cudaErrorSymbolNotFound = 500, 
# 756
cudaErrorNotReady = 600, 
# 764
cudaErrorIllegalAddress = 700, 
# 773 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorLaunchOutOfResources, 
# 784 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorLaunchTimeout, 
# 790
cudaErrorLaunchIncompatibleTexturing, 
# 797
cudaErrorPeerAccessAlreadyEnabled, 
# 804
cudaErrorPeerAccessNotEnabled, 
# 817 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorSetOnActiveProcess = 708, 
# 824
cudaErrorContextIsDestroyed, 
# 831
cudaErrorAssert, 
# 838
cudaErrorTooManyPeers, 
# 844
cudaErrorHostMemoryAlreadyRegistered, 
# 850
cudaErrorHostMemoryNotRegistered, 
# 859 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorHardwareStackError, 
# 867
cudaErrorIllegalInstruction, 
# 876 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorMisalignedAddress, 
# 887 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorInvalidAddressSpace, 
# 895
cudaErrorInvalidPc, 
# 906 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorLaunchFailure, 
# 915 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorCooperativeLaunchTooLarge, 
# 920
cudaErrorNotPermitted = 800, 
# 926
cudaErrorNotSupported, 
# 935 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorSystemNotReady, 
# 942
cudaErrorSystemDriverMismatch, 
# 951 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorCompatNotSupportedOnDevice, 
# 956
cudaErrorMpsConnectionFailed, 
# 961
cudaErrorMpsRpcFailure, 
# 967
cudaErrorMpsServerNotReady, 
# 972
cudaErrorMpsMaxClientsReached, 
# 977
cudaErrorMpsMaxConnectionsReached, 
# 982
cudaErrorMpsClientTerminated, 
# 987
cudaErrorStreamCaptureUnsupported = 900, 
# 993
cudaErrorStreamCaptureInvalidated, 
# 999
cudaErrorStreamCaptureMerge, 
# 1004
cudaErrorStreamCaptureUnmatched, 
# 1010
cudaErrorStreamCaptureUnjoined, 
# 1017
cudaErrorStreamCaptureIsolation, 
# 1023
cudaErrorStreamCaptureImplicit, 
# 1029
cudaErrorCapturedEvent, 
# 1036
cudaErrorStreamCaptureWrongThread, 
# 1041
cudaErrorTimeout, 
# 1047
cudaErrorGraphExecUpdateFailure, 
# 1057 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaErrorExternalDevice, 
# 1063
cudaErrorInvalidClusterSize, 
# 1068
cudaErrorUnknown = 999, 
# 1076
cudaErrorApiFailureBase = 10000
# 1077
}; 
#endif
# 1082 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1082
enum cudaChannelFormatKind { 
# 1084
cudaChannelFormatKindSigned, 
# 1085
cudaChannelFormatKindUnsigned, 
# 1086
cudaChannelFormatKindFloat, 
# 1087
cudaChannelFormatKindNone, 
# 1088
cudaChannelFormatKindNV12, 
# 1089
cudaChannelFormatKindUnsignedNormalized8X1, 
# 1090
cudaChannelFormatKindUnsignedNormalized8X2, 
# 1091
cudaChannelFormatKindUnsignedNormalized8X4, 
# 1092
cudaChannelFormatKindUnsignedNormalized16X1, 
# 1093
cudaChannelFormatKindUnsignedNormalized16X2, 
# 1094
cudaChannelFormatKindUnsignedNormalized16X4, 
# 1095
cudaChannelFormatKindSignedNormalized8X1, 
# 1096
cudaChannelFormatKindSignedNormalized8X2, 
# 1097
cudaChannelFormatKindSignedNormalized8X4, 
# 1098
cudaChannelFormatKindSignedNormalized16X1, 
# 1099
cudaChannelFormatKindSignedNormalized16X2, 
# 1100
cudaChannelFormatKindSignedNormalized16X4, 
# 1101
cudaChannelFormatKindUnsignedBlockCompressed1, 
# 1102
cudaChannelFormatKindUnsignedBlockCompressed1SRGB, 
# 1103
cudaChannelFormatKindUnsignedBlockCompressed2, 
# 1104
cudaChannelFormatKindUnsignedBlockCompressed2SRGB, 
# 1105
cudaChannelFormatKindUnsignedBlockCompressed3, 
# 1106
cudaChannelFormatKindUnsignedBlockCompressed3SRGB, 
# 1107
cudaChannelFormatKindUnsignedBlockCompressed4, 
# 1108
cudaChannelFormatKindSignedBlockCompressed4, 
# 1109
cudaChannelFormatKindUnsignedBlockCompressed5, 
# 1110
cudaChannelFormatKindSignedBlockCompressed5, 
# 1111
cudaChannelFormatKindUnsignedBlockCompressed6H, 
# 1112
cudaChannelFormatKindSignedBlockCompressed6H, 
# 1113
cudaChannelFormatKindUnsignedBlockCompressed7, 
# 1114
cudaChannelFormatKindUnsignedBlockCompressed7SRGB
# 1115
}; 
#endif
# 1120 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1120
struct cudaChannelFormatDesc { 
# 1122
int x; 
# 1123
int y; 
# 1124
int z; 
# 1125
int w; 
# 1126
cudaChannelFormatKind f; 
# 1127
}; 
#endif
# 1132 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
typedef struct cudaArray *cudaArray_t; 
# 1137
typedef const cudaArray *cudaArray_const_t; 
# 1139
struct cudaArray; 
# 1144
typedef struct cudaMipmappedArray *cudaMipmappedArray_t; 
# 1149
typedef const cudaMipmappedArray *cudaMipmappedArray_const_t; 
# 1151
struct cudaMipmappedArray; 
# 1161 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1161
struct cudaArraySparseProperties { 
# 1162
struct { 
# 1163
unsigned width; 
# 1164
unsigned height; 
# 1165
unsigned depth; 
# 1166
} tileExtent; 
# 1167
unsigned miptailFirstLevel; 
# 1168
unsigned long long miptailSize; 
# 1169
unsigned flags; 
# 1170
unsigned reserved[4]; 
# 1171
}; 
#endif
# 1176 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1176
struct cudaArrayMemoryRequirements { 
# 1177
size_t size; 
# 1178
size_t alignment; 
# 1179
unsigned reserved[4]; 
# 1180
}; 
#endif
# 1185 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1185
enum cudaMemoryType { 
# 1187
cudaMemoryTypeUnregistered, 
# 1188
cudaMemoryTypeHost, 
# 1189
cudaMemoryTypeDevice, 
# 1190
cudaMemoryTypeManaged
# 1191
}; 
#endif
# 1196 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1196
enum cudaMemcpyKind { 
# 1198
cudaMemcpyHostToHost, 
# 1199
cudaMemcpyHostToDevice, 
# 1200
cudaMemcpyDeviceToHost, 
# 1201
cudaMemcpyDeviceToDevice, 
# 1202
cudaMemcpyDefault
# 1203
}; 
#endif
# 1210 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1210
struct cudaPitchedPtr { 
# 1212
void *ptr; 
# 1213
size_t pitch; 
# 1214
size_t xsize; 
# 1215
size_t ysize; 
# 1216
}; 
#endif
# 1223 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1223
struct cudaExtent { 
# 1225
size_t width; 
# 1226
size_t height; 
# 1227
size_t depth; 
# 1228
}; 
#endif
# 1235 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1235
struct cudaPos { 
# 1237
size_t x; 
# 1238
size_t y; 
# 1239
size_t z; 
# 1240
}; 
#endif
# 1245 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1245
struct cudaMemcpy3DParms { 
# 1247
cudaArray_t srcArray; 
# 1248
cudaPos srcPos; 
# 1249
cudaPitchedPtr srcPtr; 
# 1251
cudaArray_t dstArray; 
# 1252
cudaPos dstPos; 
# 1253
cudaPitchedPtr dstPtr; 
# 1255
cudaExtent extent; 
# 1256
cudaMemcpyKind kind; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 1257
}; 
#endif
# 1262 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1262
struct cudaMemcpy3DPeerParms { 
# 1264
cudaArray_t srcArray; 
# 1265
cudaPos srcPos; 
# 1266
cudaPitchedPtr srcPtr; 
# 1267
int srcDevice; 
# 1269
cudaArray_t dstArray; 
# 1270
cudaPos dstPos; 
# 1271
cudaPitchedPtr dstPtr; 
# 1272
int dstDevice; 
# 1274
cudaExtent extent; 
# 1275
}; 
#endif
# 1280 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1280
struct cudaMemsetParams { 
# 1281
void *dst; 
# 1282
size_t pitch; 
# 1283
unsigned value; 
# 1284
unsigned elementSize; 
# 1285
size_t width; 
# 1286
size_t height; 
# 1287
}; 
#endif
# 1292 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1292
enum cudaAccessProperty { 
# 1293
cudaAccessPropertyNormal, 
# 1294
cudaAccessPropertyStreaming, 
# 1295
cudaAccessPropertyPersisting
# 1296
}; 
#endif
# 1309 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1309
struct cudaAccessPolicyWindow { 
# 1310
void *base_ptr; 
# 1311
size_t num_bytes; 
# 1312
float hitRatio; 
# 1313
cudaAccessProperty hitProp; 
# 1314
cudaAccessProperty missProp; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 1315
}; 
#endif
# 1327 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
typedef void (*cudaHostFn_t)(void * userData); 
# 1332
#if 0
# 1332
struct cudaHostNodeParams { 
# 1333
cudaHostFn_t fn; 
# 1334
void *userData; 
# 1335
}; 
#endif
# 1340 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1340
enum cudaStreamCaptureStatus { 
# 1341
cudaStreamCaptureStatusNone, 
# 1342
cudaStreamCaptureStatusActive, 
# 1343
cudaStreamCaptureStatusInvalidated
# 1345
}; 
#endif
# 1351 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1351
enum cudaStreamCaptureMode { 
# 1352
cudaStreamCaptureModeGlobal, 
# 1353
cudaStreamCaptureModeThreadLocal, 
# 1354
cudaStreamCaptureModeRelaxed
# 1355
}; 
#endif
# 1357 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1357
enum cudaSynchronizationPolicy { 
# 1358
cudaSyncPolicyAuto = 1, 
# 1359
cudaSyncPolicySpin, 
# 1360
cudaSyncPolicyYield, 
# 1361
cudaSyncPolicyBlockingSync
# 1362
}; 
#endif
# 1367 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1367
enum cudaClusterSchedulingPolicy { 
# 1368
cudaClusterSchedulingPolicyDefault, 
# 1369
cudaClusterSchedulingPolicySpread, 
# 1370
cudaClusterSchedulingPolicyLoadBalancing
# 1371
}; 
#endif
# 1376 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1376
enum cudaStreamUpdateCaptureDependenciesFlags { 
# 1377
cudaStreamAddCaptureDependencies, 
# 1378
cudaStreamSetCaptureDependencies
# 1379
}; 
#endif
# 1384 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1384
enum cudaUserObjectFlags { 
# 1385
cudaUserObjectNoDestructorSync = 1
# 1386
}; 
#endif
# 1391 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1391
enum cudaUserObjectRetainFlags { 
# 1392
cudaGraphUserObjectMove = 1
# 1393
}; 
#endif
# 1398 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
struct cudaGraphicsResource; 
# 1403
#if 0
# 1403
enum cudaGraphicsRegisterFlags { 
# 1405
cudaGraphicsRegisterFlagsNone, 
# 1406
cudaGraphicsRegisterFlagsReadOnly, 
# 1407
cudaGraphicsRegisterFlagsWriteDiscard, 
# 1408
cudaGraphicsRegisterFlagsSurfaceLoadStore = 4, 
# 1409
cudaGraphicsRegisterFlagsTextureGather = 8
# 1410
}; 
#endif
# 1415 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1415
enum cudaGraphicsMapFlags { 
# 1417
cudaGraphicsMapFlagsNone, 
# 1418
cudaGraphicsMapFlagsReadOnly, 
# 1419
cudaGraphicsMapFlagsWriteDiscard
# 1420
}; 
#endif
# 1425 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1425
enum cudaGraphicsCubeFace { 
# 1427
cudaGraphicsCubeFacePositiveX, 
# 1428
cudaGraphicsCubeFaceNegativeX, 
# 1429
cudaGraphicsCubeFacePositiveY, 
# 1430
cudaGraphicsCubeFaceNegativeY, 
# 1431
cudaGraphicsCubeFacePositiveZ, 
# 1432
cudaGraphicsCubeFaceNegativeZ
# 1433
}; 
#endif
# 1438 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1438
enum cudaResourceType { 
# 1440
cudaResourceTypeArray, 
# 1441
cudaResourceTypeMipmappedArray, 
# 1442
cudaResourceTypeLinear, 
# 1443
cudaResourceTypePitch2D
# 1444
}; 
#endif
# 1449 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1449
enum cudaResourceViewFormat { 
# 1451
cudaResViewFormatNone, 
# 1452
cudaResViewFormatUnsignedChar1, 
# 1453
cudaResViewFormatUnsignedChar2, 
# 1454
cudaResViewFormatUnsignedChar4, 
# 1455
cudaResViewFormatSignedChar1, 
# 1456
cudaResViewFormatSignedChar2, 
# 1457
cudaResViewFormatSignedChar4, 
# 1458
cudaResViewFormatUnsignedShort1, 
# 1459
cudaResViewFormatUnsignedShort2, 
# 1460
cudaResViewFormatUnsignedShort4, 
# 1461
cudaResViewFormatSignedShort1, 
# 1462
cudaResViewFormatSignedShort2, 
# 1463
cudaResViewFormatSignedShort4, 
# 1464
cudaResViewFormatUnsignedInt1, 
# 1465
cudaResViewFormatUnsignedInt2, 
# 1466
cudaResViewFormatUnsignedInt4, 
# 1467
cudaResViewFormatSignedInt1, 
# 1468
cudaResViewFormatSignedInt2, 
# 1469
cudaResViewFormatSignedInt4, 
# 1470
cudaResViewFormatHalf1, 
# 1471
cudaResViewFormatHalf2, 
# 1472
cudaResViewFormatHalf4, 
# 1473
cudaResViewFormatFloat1, 
# 1474
cudaResViewFormatFloat2, 
# 1475
cudaResViewFormatFloat4, 
# 1476
cudaResViewFormatUnsignedBlockCompressed1, 
# 1477
cudaResViewFormatUnsignedBlockCompressed2, 
# 1478
cudaResViewFormatUnsignedBlockCompressed3, 
# 1479
cudaResViewFormatUnsignedBlockCompressed4, 
# 1480
cudaResViewFormatSignedBlockCompressed4, 
# 1481
cudaResViewFormatUnsignedBlockCompressed5, 
# 1482
cudaResViewFormatSignedBlockCompressed5, 
# 1483
cudaResViewFormatUnsignedBlockCompressed6H, 
# 1484
cudaResViewFormatSignedBlockCompressed6H, 
# 1485
cudaResViewFormatUnsignedBlockCompressed7
# 1486
}; 
#endif
# 1491 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1491
struct cudaResourceDesc { 
# 1492
cudaResourceType resType; 
# 1494
union { 
# 1495
struct { 
# 1496
cudaArray_t array; 
# 1497
} array; 
# 1498
struct { 
# 1499
cudaMipmappedArray_t mipmap; 
# 1500
} mipmap; 
# 1501
struct { 
# 1502
void *devPtr; 
# 1503
cudaChannelFormatDesc desc; 
# 1504
size_t sizeInBytes; 
# 1505
} linear; 
# 1506
struct { 
# 1507
void *devPtr; 
# 1508
cudaChannelFormatDesc desc; 
# 1509
size_t width; 
# 1510
size_t height; 
# 1511
size_t pitchInBytes; 
# 1512
} pitch2D; 
# 1513
} res; 
# 1514
}; 
#endif
# 1519 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1519
struct cudaResourceViewDesc { 
# 1521
cudaResourceViewFormat format; 
# 1522
size_t width; 
# 1523
size_t height; 
# 1524
size_t depth; 
# 1525
unsigned firstMipmapLevel; 
# 1526
unsigned lastMipmapLevel; 
# 1527
unsigned firstLayer; 
# 1528
unsigned lastLayer; 
# 1529
}; 
#endif
# 1534 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1534
struct cudaPointerAttributes { 
# 1540
cudaMemoryType type; 
# 1551 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
int device; 
# 1557
void *devicePointer; 
# 1566 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
void *hostPointer; 
# 1567
}; 
#endif
# 1572 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1572
struct cudaFuncAttributes { 
# 1579
size_t sharedSizeBytes; 
# 1585
size_t constSizeBytes; 
# 1590
size_t localSizeBytes; 
# 1597
int maxThreadsPerBlock; 
# 1602
int numRegs; 
# 1609
int ptxVersion; 
# 1616
int binaryVersion; 
# 1622
int cacheModeCA; 
# 1629
int maxDynamicSharedSizeBytes; 
# 1638 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
int preferredShmemCarveout; 
# 1639
}; 
#endif
# 1644 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1644
enum cudaFuncAttribute { 
# 1646
cudaFuncAttributeMaxDynamicSharedMemorySize = 8, 
# 1647
cudaFuncAttributePreferredSharedMemoryCarveout, 
# 1648
cudaFuncAttributeClusterDimMustBeSet, 
# 1649
cudaFuncAttributeRequiredClusterWidth, 
# 1650
cudaFuncAttributeRequiredClusterHeight, 
# 1651
cudaFuncAttributeRequiredClusterDepth, 
# 1652
cudaFuncAttributeNonPortableClusterSizeAllowed, 
# 1653
cudaFuncAttributeClusterSchedulingPolicyPreference, 
# 1654
cudaFuncAttributeMax
# 1655
}; 
#endif
# 1660 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1660
enum cudaFuncCache { 
# 1662
cudaFuncCachePreferNone, 
# 1663
cudaFuncCachePreferShared, 
# 1664
cudaFuncCachePreferL1, 
# 1665
cudaFuncCachePreferEqual
# 1666
}; 
#endif
# 1672 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1672
enum cudaSharedMemConfig { 
# 1674
cudaSharedMemBankSizeDefault, 
# 1675
cudaSharedMemBankSizeFourByte, 
# 1676
cudaSharedMemBankSizeEightByte
# 1677
}; 
#endif
# 1682 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1682
enum cudaSharedCarveout { 
# 1683
cudaSharedmemCarveoutDefault = (-1), 
# 1684
cudaSharedmemCarveoutMaxShared = 100, 
# 1685
cudaSharedmemCarveoutMaxL1 = 0
# 1686
}; 
#endif
# 1691 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1691
enum cudaComputeMode { 
# 1693
cudaComputeModeDefault, 
# 1694
cudaComputeModeExclusive, 
# 1695
cudaComputeModeProhibited, 
# 1696
cudaComputeModeExclusiveProcess
# 1697
}; 
#endif
# 1702 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1702
enum cudaLimit { 
# 1704
cudaLimitStackSize, 
# 1705
cudaLimitPrintfFifoSize, 
# 1706
cudaLimitMallocHeapSize, 
# 1707
cudaLimitDevRuntimeSyncDepth, 
# 1708
cudaLimitDevRuntimePendingLaunchCount, 
# 1709
cudaLimitMaxL2FetchGranularity, 
# 1710
cudaLimitPersistingL2CacheSize
# 1711
}; 
#endif
# 1716 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1716
enum cudaMemoryAdvise { 
# 1718
cudaMemAdviseSetReadMostly = 1, 
# 1719
cudaMemAdviseUnsetReadMostly, 
# 1720
cudaMemAdviseSetPreferredLocation, 
# 1721
cudaMemAdviseUnsetPreferredLocation, 
# 1722
cudaMemAdviseSetAccessedBy, 
# 1723
cudaMemAdviseUnsetAccessedBy
# 1724
}; 
#endif
# 1729 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1729
enum cudaMemRangeAttribute { 
# 1731
cudaMemRangeAttributeReadMostly = 1, 
# 1732
cudaMemRangeAttributePreferredLocation, 
# 1733
cudaMemRangeAttributeAccessedBy, 
# 1734
cudaMemRangeAttributeLastPrefetchLocation
# 1735
}; 
#endif
# 1740 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1740
enum cudaOutputMode { 
# 1742
cudaKeyValuePair, 
# 1743
cudaCSV
# 1744
}; 
#endif
# 1749 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1749
enum cudaFlushGPUDirectRDMAWritesOptions { 
# 1750
cudaFlushGPUDirectRDMAWritesOptionHost = (1 << 0), 
# 1751
cudaFlushGPUDirectRDMAWritesOptionMemOps
# 1752
}; 
#endif
# 1757 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1757
enum cudaGPUDirectRDMAWritesOrdering { 
# 1758
cudaGPUDirectRDMAWritesOrderingNone, 
# 1759
cudaGPUDirectRDMAWritesOrderingOwner = 100, 
# 1760
cudaGPUDirectRDMAWritesOrderingAllDevices = 200
# 1761
}; 
#endif
# 1766 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1766
enum cudaFlushGPUDirectRDMAWritesScope { 
# 1767
cudaFlushGPUDirectRDMAWritesToOwner = 100, 
# 1768
cudaFlushGPUDirectRDMAWritesToAllDevices = 200
# 1769
}; 
#endif
# 1774 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1774
enum cudaFlushGPUDirectRDMAWritesTarget { 
# 1775
cudaFlushGPUDirectRDMAWritesTargetCurrentDevice
# 1776
}; 
#endif
# 1782 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1782
enum cudaDeviceAttr { 
# 1784
cudaDevAttrMaxThreadsPerBlock = 1, 
# 1785
cudaDevAttrMaxBlockDimX, 
# 1786
cudaDevAttrMaxBlockDimY, 
# 1787
cudaDevAttrMaxBlockDimZ, 
# 1788
cudaDevAttrMaxGridDimX, 
# 1789
cudaDevAttrMaxGridDimY, 
# 1790
cudaDevAttrMaxGridDimZ, 
# 1791
cudaDevAttrMaxSharedMemoryPerBlock, 
# 1792
cudaDevAttrTotalConstantMemory, 
# 1793
cudaDevAttrWarpSize, 
# 1794
cudaDevAttrMaxPitch, 
# 1795
cudaDevAttrMaxRegistersPerBlock, 
# 1796
cudaDevAttrClockRate, 
# 1797
cudaDevAttrTextureAlignment, 
# 1798
cudaDevAttrGpuOverlap, 
# 1799
cudaDevAttrMultiProcessorCount, 
# 1800
cudaDevAttrKernelExecTimeout, 
# 1801
cudaDevAttrIntegrated, 
# 1802
cudaDevAttrCanMapHostMemory, 
# 1803
cudaDevAttrComputeMode, 
# 1804
cudaDevAttrMaxTexture1DWidth, 
# 1805
cudaDevAttrMaxTexture2DWidth, 
# 1806
cudaDevAttrMaxTexture2DHeight, 
# 1807
cudaDevAttrMaxTexture3DWidth, 
# 1808
cudaDevAttrMaxTexture3DHeight, 
# 1809
cudaDevAttrMaxTexture3DDepth, 
# 1810
cudaDevAttrMaxTexture2DLayeredWidth, 
# 1811
cudaDevAttrMaxTexture2DLayeredHeight, 
# 1812
cudaDevAttrMaxTexture2DLayeredLayers, 
# 1813
cudaDevAttrSurfaceAlignment, 
# 1814
cudaDevAttrConcurrentKernels, 
# 1815
cudaDevAttrEccEnabled, 
# 1816
cudaDevAttrPciBusId, 
# 1817
cudaDevAttrPciDeviceId, 
# 1818
cudaDevAttrTccDriver, 
# 1819
cudaDevAttrMemoryClockRate, 
# 1820
cudaDevAttrGlobalMemoryBusWidth, 
# 1821
cudaDevAttrL2CacheSize, 
# 1822
cudaDevAttrMaxThreadsPerMultiProcessor, 
# 1823
cudaDevAttrAsyncEngineCount, 
# 1824
cudaDevAttrUnifiedAddressing, 
# 1825
cudaDevAttrMaxTexture1DLayeredWidth, 
# 1826
cudaDevAttrMaxTexture1DLayeredLayers, 
# 1827
cudaDevAttrMaxTexture2DGatherWidth = 45, 
# 1828
cudaDevAttrMaxTexture2DGatherHeight, 
# 1829
cudaDevAttrMaxTexture3DWidthAlt, 
# 1830
cudaDevAttrMaxTexture3DHeightAlt, 
# 1831
cudaDevAttrMaxTexture3DDepthAlt, 
# 1832
cudaDevAttrPciDomainId, 
# 1833
cudaDevAttrTexturePitchAlignment, 
# 1834
cudaDevAttrMaxTextureCubemapWidth, 
# 1835
cudaDevAttrMaxTextureCubemapLayeredWidth, 
# 1836
cudaDevAttrMaxTextureCubemapLayeredLayers, 
# 1837
cudaDevAttrMaxSurface1DWidth, 
# 1838
cudaDevAttrMaxSurface2DWidth, 
# 1839
cudaDevAttrMaxSurface2DHeight, 
# 1840
cudaDevAttrMaxSurface3DWidth, 
# 1841
cudaDevAttrMaxSurface3DHeight, 
# 1842
cudaDevAttrMaxSurface3DDepth, 
# 1843
cudaDevAttrMaxSurface1DLayeredWidth, 
# 1844
cudaDevAttrMaxSurface1DLayeredLayers, 
# 1845
cudaDevAttrMaxSurface2DLayeredWidth, 
# 1846
cudaDevAttrMaxSurface2DLayeredHeight, 
# 1847
cudaDevAttrMaxSurface2DLayeredLayers, 
# 1848
cudaDevAttrMaxSurfaceCubemapWidth, 
# 1849
cudaDevAttrMaxSurfaceCubemapLayeredWidth, 
# 1850
cudaDevAttrMaxSurfaceCubemapLayeredLayers, 
# 1851
cudaDevAttrMaxTexture1DLinearWidth, 
# 1852
cudaDevAttrMaxTexture2DLinearWidth, 
# 1853
cudaDevAttrMaxTexture2DLinearHeight, 
# 1854
cudaDevAttrMaxTexture2DLinearPitch, 
# 1855
cudaDevAttrMaxTexture2DMipmappedWidth, 
# 1856
cudaDevAttrMaxTexture2DMipmappedHeight, 
# 1857
cudaDevAttrComputeCapabilityMajor, 
# 1858
cudaDevAttrComputeCapabilityMinor, 
# 1859
cudaDevAttrMaxTexture1DMipmappedWidth, 
# 1860
cudaDevAttrStreamPrioritiesSupported, 
# 1861
cudaDevAttrGlobalL1CacheSupported, 
# 1862
cudaDevAttrLocalL1CacheSupported, 
# 1863
cudaDevAttrMaxSharedMemoryPerMultiprocessor, 
# 1864
cudaDevAttrMaxRegistersPerMultiprocessor, 
# 1865
cudaDevAttrManagedMemory, 
# 1866
cudaDevAttrIsMultiGpuBoard, 
# 1867
cudaDevAttrMultiGpuBoardGroupID, 
# 1868
cudaDevAttrHostNativeAtomicSupported, 
# 1869
cudaDevAttrSingleToDoublePrecisionPerfRatio, 
# 1870
cudaDevAttrPageableMemoryAccess, 
# 1871
cudaDevAttrConcurrentManagedAccess, 
# 1872
cudaDevAttrComputePreemptionSupported, 
# 1873
cudaDevAttrCanUseHostPointerForRegisteredMem, 
# 1874
cudaDevAttrReserved92, 
# 1875
cudaDevAttrReserved93, 
# 1876
cudaDevAttrReserved94, 
# 1877
cudaDevAttrCooperativeLaunch, 
# 1878
cudaDevAttrCooperativeMultiDeviceLaunch, 
# 1879
cudaDevAttrMaxSharedMemoryPerBlockOptin, 
# 1880
cudaDevAttrCanFlushRemoteWrites, 
# 1881
cudaDevAttrHostRegisterSupported, 
# 1882
cudaDevAttrPageableMemoryAccessUsesHostPageTables, 
# 1883
cudaDevAttrDirectManagedMemAccessFromHost, 
# 1884
cudaDevAttrMaxBlocksPerMultiprocessor = 106, 
# 1885
cudaDevAttrMaxPersistingL2CacheSize = 108, 
# 1886
cudaDevAttrMaxAccessPolicyWindowSize, 
# 1887
cudaDevAttrReservedSharedMemoryPerBlock = 111, 
# 1888
cudaDevAttrSparseCudaArraySupported, 
# 1889
cudaDevAttrHostRegisterReadOnlySupported, 
# 1890
cudaDevAttrTimelineSemaphoreInteropSupported, 
# 1891
cudaDevAttrMaxTimelineSemaphoreInteropSupported = 114, 
# 1892
cudaDevAttrMemoryPoolsSupported, 
# 1893
cudaDevAttrGPUDirectRDMASupported, 
# 1894
cudaDevAttrGPUDirectRDMAFlushWritesOptions, 
# 1895
cudaDevAttrGPUDirectRDMAWritesOrdering, 
# 1896
cudaDevAttrMemoryPoolSupportedHandleTypes, 
# 1897
cudaDevAttrClusterLaunch, 
# 1898
cudaDevAttrDeferredMappingCudaArraySupported, 
# 1899
cudaDevAttrMax
# 1900
}; 
#endif
# 1905 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1905
enum cudaMemPoolAttr { 
# 1915 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaMemPoolReuseFollowEventDependencies = 1, 
# 1922
cudaMemPoolReuseAllowOpportunistic, 
# 1930
cudaMemPoolReuseAllowInternalDependencies, 
# 1941 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaMemPoolAttrReleaseThreshold, 
# 1947
cudaMemPoolAttrReservedMemCurrent, 
# 1954
cudaMemPoolAttrReservedMemHigh, 
# 1960
cudaMemPoolAttrUsedMemCurrent, 
# 1967
cudaMemPoolAttrUsedMemHigh
# 1968
}; 
#endif
# 1973 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1973
enum cudaMemLocationType { 
# 1974
cudaMemLocationTypeInvalid, 
# 1975
cudaMemLocationTypeDevice
# 1976
}; 
#endif
# 1983 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1983
struct cudaMemLocation { 
# 1984
cudaMemLocationType type; 
# 1985
int id; 
# 1986
}; 
#endif
# 1991 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 1991
enum cudaMemAccessFlags { 
# 1992
cudaMemAccessFlagsProtNone, 
# 1993
cudaMemAccessFlagsProtRead, 
# 1994
cudaMemAccessFlagsProtReadWrite = 3
# 1995
}; 
#endif
# 2000 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2000
struct cudaMemAccessDesc { 
# 2001
cudaMemLocation location; 
# 2002
cudaMemAccessFlags flags; 
# 2003
}; 
#endif
# 2008 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2008
enum cudaMemAllocationType { 
# 2009
cudaMemAllocationTypeInvalid, 
# 2013
cudaMemAllocationTypePinned, 
# 2014
cudaMemAllocationTypeMax = 2147483647
# 2015
}; 
#endif
# 2020 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2020
enum cudaMemAllocationHandleType { 
# 2021
cudaMemHandleTypeNone, 
# 2022
cudaMemHandleTypePosixFileDescriptor, 
# 2023
cudaMemHandleTypeWin32, 
# 2024
cudaMemHandleTypeWin32Kmt = 4
# 2025
}; 
#endif
# 2030 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2030
struct cudaMemPoolProps { 
# 2031
cudaMemAllocationType allocType; 
# 2032
cudaMemAllocationHandleType handleTypes; 
# 2033
cudaMemLocation location; 
# 2040
void *win32SecurityAttributes; 
# 2041
unsigned char reserved[64]; 
# 2042
}; 
#endif
# 2047 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2047
struct cudaMemPoolPtrExportData { 
# 2048
unsigned char reserved[64]; 
# 2049
}; 
#endif
# 2054 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2054
struct cudaMemAllocNodeParams { 
# 2059
cudaMemPoolProps poolProps; 
# 2060
const cudaMemAccessDesc *accessDescs; 
# 2061
size_t accessDescCount; 
# 2062
size_t bytesize; 
# 2063
void *dptr; 
# 2064
}; 
#endif
# 2069 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2069
enum cudaGraphMemAttributeType { 
# 2074
cudaGraphMemAttrUsedMemCurrent, 
# 2081
cudaGraphMemAttrUsedMemHigh, 
# 2088
cudaGraphMemAttrReservedMemCurrent, 
# 2095
cudaGraphMemAttrReservedMemHigh
# 2096
}; 
#endif
# 2102 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2102
enum cudaDeviceP2PAttr { 
# 2103
cudaDevP2PAttrPerformanceRank = 1, 
# 2104
cudaDevP2PAttrAccessSupported, 
# 2105
cudaDevP2PAttrNativeAtomicSupported, 
# 2106
cudaDevP2PAttrCudaArrayAccessSupported
# 2107
}; 
#endif
# 2114 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2114
struct CUuuid_st { 
# 2115
char bytes[16]; 
# 2116
}; 
#endif
# 2117 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
typedef CUuuid_st 
# 2117
CUuuid; 
#endif
# 2119 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
typedef CUuuid_st 
# 2119
cudaUUID_t; 
#endif
# 2124 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2124
struct cudaDeviceProp { 
# 2126
char name[256]; 
# 2127
cudaUUID_t uuid; 
# 2128
char luid[8]; 
# 2129
unsigned luidDeviceNodeMask; 
# 2130
size_t totalGlobalMem; 
# 2131
size_t sharedMemPerBlock; 
# 2132
int regsPerBlock; 
# 2133
int warpSize; 
# 2134
size_t memPitch; 
# 2135
int maxThreadsPerBlock; 
# 2136
int maxThreadsDim[3]; 
# 2137
int maxGridSize[3]; 
# 2138
int clockRate; 
# 2139
size_t totalConstMem; 
# 2140
int major; 
# 2141
int minor; 
# 2142
size_t textureAlignment; 
# 2143
size_t texturePitchAlignment; 
# 2144
int deviceOverlap; 
# 2145
int multiProcessorCount; 
# 2146
int kernelExecTimeoutEnabled; 
# 2147
int integrated; 
# 2148
int canMapHostMemory; 
# 2149
int computeMode; 
# 2150
int maxTexture1D; 
# 2151
int maxTexture1DMipmap; 
# 2152
int maxTexture1DLinear; 
# 2153
int maxTexture2D[2]; 
# 2154
int maxTexture2DMipmap[2]; 
# 2155
int maxTexture2DLinear[3]; 
# 2156
int maxTexture2DGather[2]; 
# 2157
int maxTexture3D[3]; 
# 2158
int maxTexture3DAlt[3]; 
# 2159
int maxTextureCubemap; 
# 2160
int maxTexture1DLayered[2]; 
# 2161
int maxTexture2DLayered[3]; 
# 2162
int maxTextureCubemapLayered[2]; 
# 2163
int maxSurface1D; 
# 2164
int maxSurface2D[2]; 
# 2165
int maxSurface3D[3]; 
# 2166
int maxSurface1DLayered[2]; 
# 2167
int maxSurface2DLayered[3]; 
# 2168
int maxSurfaceCubemap; 
# 2169
int maxSurfaceCubemapLayered[2]; 
# 2170
size_t surfaceAlignment; 
# 2171
int concurrentKernels; 
# 2172
int ECCEnabled; 
# 2173
int pciBusID; 
# 2174
int pciDeviceID; 
# 2175
int pciDomainID; 
# 2176
int tccDriver; 
# 2177
int asyncEngineCount; 
# 2178
int unifiedAddressing; 
# 2179
int memoryClockRate; 
# 2180
int memoryBusWidth; 
# 2181
int l2CacheSize; 
# 2182
int persistingL2CacheMaxSize; 
# 2183
int maxThreadsPerMultiProcessor; 
# 2184
int streamPrioritiesSupported; 
# 2185
int globalL1CacheSupported; 
# 2186
int localL1CacheSupported; 
# 2187
size_t sharedMemPerMultiprocessor; 
# 2188
int regsPerMultiprocessor; 
# 2189
int managedMemory; 
# 2190
int isMultiGpuBoard; 
# 2191
int multiGpuBoardGroupID; 
# 2192
int hostNativeAtomicSupported; 
# 2193
int singleToDoublePrecisionPerfRatio; 
# 2194
int pageableMemoryAccess; 
# 2195
int concurrentManagedAccess; 
# 2196
int computePreemptionSupported; 
# 2197
int canUseHostPointerForRegisteredMem; 
# 2198
int cooperativeLaunch; 
# 2199
int cooperativeMultiDeviceLaunch; 
# 2200
size_t sharedMemPerBlockOptin; 
# 2201
int pageableMemoryAccessUsesHostPageTables; 
# 2202
int directManagedMemAccessFromHost; 
# 2203
int maxBlocksPerMultiProcessor; 
# 2204
int accessPolicyMaxWindowSize; 
# 2205
size_t reservedSharedMemPerBlock; 
# 2206
}; 
#endif
# 2302 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
typedef 
# 2299
struct cudaIpcEventHandle_st { 
# 2301
char reserved[64]; 
# 2302
} cudaIpcEventHandle_t; 
#endif
# 2310 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
typedef 
# 2307
struct cudaIpcMemHandle_st { 
# 2309
char reserved[64]; 
# 2310
} cudaIpcMemHandle_t; 
#endif
# 2315 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2315
enum cudaExternalMemoryHandleType { 
# 2319
cudaExternalMemoryHandleTypeOpaqueFd = 1, 
# 2323
cudaExternalMemoryHandleTypeOpaqueWin32, 
# 2327
cudaExternalMemoryHandleTypeOpaqueWin32Kmt, 
# 2331
cudaExternalMemoryHandleTypeD3D12Heap, 
# 2335
cudaExternalMemoryHandleTypeD3D12Resource, 
# 2339
cudaExternalMemoryHandleTypeD3D11Resource, 
# 2343
cudaExternalMemoryHandleTypeD3D11ResourceKmt, 
# 2347
cudaExternalMemoryHandleTypeNvSciBuf
# 2348
}; 
#endif
# 2390 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2390
struct cudaExternalMemoryHandleDesc { 
# 2394
cudaExternalMemoryHandleType type; 
# 2395
union { 
# 2401
int fd; 
# 2417 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
struct { 
# 2421
void *handle; 
# 2426
const void *name; 
# 2427
} win32; 
# 2432
const void *nvSciBufObject; 
# 2433
} handle; 
# 2437
unsigned long long size; 
# 2441
unsigned flags; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 2442
}; 
#endif
# 2447 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2447
struct cudaExternalMemoryBufferDesc { 
# 2451
unsigned long long offset; 
# 2455
unsigned long long size; 
# 2459
unsigned flags; 
# 2460
}; 
#endif
# 2465 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2465
struct cudaExternalMemoryMipmappedArrayDesc { 
# 2470
unsigned long long offset; 
# 2474
cudaChannelFormatDesc formatDesc; 
# 2478
cudaExtent extent; 
# 2483
unsigned flags; 
# 2487
unsigned numLevels; 
# 2488
}; 
#endif
# 2493 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2493
enum cudaExternalSemaphoreHandleType { 
# 2497
cudaExternalSemaphoreHandleTypeOpaqueFd = 1, 
# 2501
cudaExternalSemaphoreHandleTypeOpaqueWin32, 
# 2505
cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt, 
# 2509
cudaExternalSemaphoreHandleTypeD3D12Fence, 
# 2513
cudaExternalSemaphoreHandleTypeD3D11Fence, 
# 2517
cudaExternalSemaphoreHandleTypeNvSciSync, 
# 2521
cudaExternalSemaphoreHandleTypeKeyedMutex, 
# 2525
cudaExternalSemaphoreHandleTypeKeyedMutexKmt, 
# 2529
cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd, 
# 2533
cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32
# 2534
}; 
#endif
# 2539 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2539
struct cudaExternalSemaphoreHandleDesc { 
# 2543
cudaExternalSemaphoreHandleType type; 
# 2544
union { 
# 2551
int fd; 
# 2567 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
struct { 
# 2571
void *handle; 
# 2576
const void *name; 
# 2577
} win32; 
# 2581
const void *nvSciSyncObj; 
# 2582
} handle; 
# 2586
unsigned flags; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 2587
}; 
#endif
# 2592 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2592
struct cudaExternalSemaphoreSignalParams_v1 { 
# 2593
struct { 
# 2597
struct { 
# 2601
unsigned long long value; 
# 2602
} fence; 
# 2603
union { 
# 2608
void *fence; 
# 2609
unsigned long long reserved; 
# 2610
} nvSciSync; 
# 2614
struct { 
# 2618
unsigned long long key; 
# 2619
} keyedMutex; 
# 2620
} params; 
# 2631 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
unsigned flags; 
# 2632
}; 
#endif
# 2637 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2637
struct cudaExternalSemaphoreWaitParams_v1 { 
# 2638
struct { 
# 2642
struct { 
# 2646
unsigned long long value; 
# 2647
} fence; 
# 2648
union { 
# 2653
void *fence; 
# 2654
unsigned long long reserved; 
# 2655
} nvSciSync; 
# 2659
struct { 
# 2663
unsigned long long key; 
# 2667
unsigned timeoutMs; 
# 2668
} keyedMutex; 
# 2669
} params; 
# 2680 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
unsigned flags; 
# 2681
}; 
#endif
# 2686 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2686
struct cudaExternalSemaphoreSignalParams { 
# 2687
struct { 
# 2691
struct { 
# 2695
unsigned long long value; 
# 2696
} fence; 
# 2697
union { 
# 2702
void *fence; 
# 2703
unsigned long long reserved; 
# 2704
} nvSciSync; 
# 2708
struct { 
# 2712
unsigned long long key; 
# 2713
} keyedMutex; 
# 2714
unsigned reserved[12]; 
# 2715
} params; 
# 2726 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
unsigned flags; 
# 2727
unsigned reserved[16]; 
# 2728
}; 
#endif
# 2733 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2733
struct cudaExternalSemaphoreWaitParams { 
# 2734
struct { 
# 2738
struct { 
# 2742
unsigned long long value; 
# 2743
} fence; 
# 2744
union { 
# 2749
void *fence; 
# 2750
unsigned long long reserved; 
# 2751
} nvSciSync; 
# 2755
struct { 
# 2759
unsigned long long key; 
# 2763
unsigned timeoutMs; 
# 2764
} keyedMutex; 
# 2765
unsigned reserved[10]; 
# 2766
} params; 
# 2777 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
unsigned flags; 
# 2778
unsigned reserved[16]; 
# 2779
}; 
#endif
# 2790 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
typedef cudaError 
# 2790
cudaError_t; 
#endif
# 2795 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
typedef struct CUstream_st *
# 2795
cudaStream_t; 
#endif
# 2800 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
typedef struct CUevent_st *
# 2800
cudaEvent_t; 
#endif
# 2805 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
typedef cudaGraphicsResource *
# 2805
cudaGraphicsResource_t; 
#endif
# 2810 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
typedef cudaOutputMode 
# 2810
cudaOutputMode_t; 
#endif
# 2815 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
typedef struct CUexternalMemory_st *
# 2815
cudaExternalMemory_t; 
#endif
# 2820 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
typedef struct CUexternalSemaphore_st *
# 2820
cudaExternalSemaphore_t; 
#endif
# 2825 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
typedef struct CUgraph_st *
# 2825
cudaGraph_t; 
#endif
# 2830 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
typedef struct CUgraphNode_st *
# 2830
cudaGraphNode_t; 
#endif
# 2835 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
typedef struct CUuserObject_st *
# 2835
cudaUserObject_t; 
#endif
# 2840 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
typedef struct CUfunc_st *
# 2840
cudaFunction_t; 
#endif
# 2845 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
typedef struct CUmemPoolHandle_st *
# 2845
cudaMemPool_t; 
#endif
# 2850 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2850
enum cudaCGScope { 
# 2851
cudaCGScopeInvalid, 
# 2852
cudaCGScopeGrid, 
# 2853
cudaCGScopeMultiGrid
# 2854
}; 
#endif
# 2859 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2859
struct cudaLaunchParams { 
# 2861
void *func; 
# 2862
dim3 gridDim; 
# 2863
dim3 blockDim; 
# 2864
void **args; 
# 2865
size_t sharedMem; 
# 2866
cudaStream_t stream; 
# 2867
}; 
#endif
# 2872 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2872
struct cudaKernelNodeParams { 
# 2873
void *func; 
# 2874
dim3 gridDim; 
# 2875
dim3 blockDim; 
# 2876
unsigned sharedMemBytes; 
# 2877
void **kernelParams; 
# 2878
void **extra; 
# 2879
}; 
#endif
# 2884 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2884
struct cudaExternalSemaphoreSignalNodeParams { 
# 2885
cudaExternalSemaphore_t *extSemArray; 
# 2886
const cudaExternalSemaphoreSignalParams *paramsArray; 
# 2887
unsigned numExtSems; 
# 2888
}; 
#endif
# 2893 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2893
struct cudaExternalSemaphoreWaitNodeParams { 
# 2894
cudaExternalSemaphore_t *extSemArray; 
# 2895
const cudaExternalSemaphoreWaitParams *paramsArray; 
# 2896
unsigned numExtSems; 
# 2897
}; 
#endif
# 2902 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2902
enum cudaGraphNodeType { 
# 2903
cudaGraphNodeTypeKernel, 
# 2904
cudaGraphNodeTypeMemcpy, 
# 2905
cudaGraphNodeTypeMemset, 
# 2906
cudaGraphNodeTypeHost, 
# 2907
cudaGraphNodeTypeGraph, 
# 2908
cudaGraphNodeTypeEmpty, 
# 2909
cudaGraphNodeTypeWaitEvent, 
# 2910
cudaGraphNodeTypeEventRecord, 
# 2911
cudaGraphNodeTypeExtSemaphoreSignal, 
# 2912
cudaGraphNodeTypeExtSemaphoreWait, 
# 2913
cudaGraphNodeTypeMemAlloc, 
# 2914
cudaGraphNodeTypeMemFree, 
# 2915
cudaGraphNodeTypeCount
# 2916
}; 
#endif
# 2921 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
typedef struct CUgraphExec_st *cudaGraphExec_t; 
# 2926
#if 0
# 2926
enum cudaGraphExecUpdateResult { 
# 2927
cudaGraphExecUpdateSuccess, 
# 2928
cudaGraphExecUpdateError, 
# 2929
cudaGraphExecUpdateErrorTopologyChanged, 
# 2930
cudaGraphExecUpdateErrorNodeTypeChanged, 
# 2931
cudaGraphExecUpdateErrorFunctionChanged, 
# 2932
cudaGraphExecUpdateErrorParametersChanged, 
# 2933
cudaGraphExecUpdateErrorNotSupported, 
# 2934
cudaGraphExecUpdateErrorUnsupportedFunctionChange, 
# 2935
cudaGraphExecUpdateErrorAttributesChanged
# 2936
}; 
#endif
# 2942 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2942
enum cudaGetDriverEntryPointFlags { 
# 2943
cudaEnableDefault, 
# 2944
cudaEnableLegacyStream, 
# 2945
cudaEnablePerThreadDefaultStream
# 2946
}; 
#endif
# 2951 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2951
enum cudaGraphDebugDotFlags { 
# 2952
cudaGraphDebugDotFlagsVerbose = (1 << 0), 
# 2953
cudaGraphDebugDotFlagsKernelNodeParams = (1 << 2), 
# 2954
cudaGraphDebugDotFlagsMemcpyNodeParams = (1 << 3), 
# 2955
cudaGraphDebugDotFlagsMemsetNodeParams = (1 << 4), 
# 2956
cudaGraphDebugDotFlagsHostNodeParams = (1 << 5), 
# 2957
cudaGraphDebugDotFlagsEventNodeParams = (1 << 6), 
# 2958
cudaGraphDebugDotFlagsExtSemasSignalNodeParams = (1 << 7), 
# 2959
cudaGraphDebugDotFlagsExtSemasWaitNodeParams = (1 << 8), 
# 2960
cudaGraphDebugDotFlagsKernelNodeAttributes = (1 << 9), 
# 2961
cudaGraphDebugDotFlagsHandles = (1 << 10)
# 2962
}; 
#endif
# 2967 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
# 2967
enum cudaGraphInstantiateFlags { 
# 2968
cudaGraphInstantiateFlagAutoFreeOnLaunch = 1, 
# 2969
cudaGraphInstantiateFlagUseNodePriority = 8
# 2971
}; 
#endif
# 3010 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
typedef 
# 2976 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
enum cudaLaunchAttributeID { 
# 2977
cudaLaunchAttributeIgnore, 
# 2978
cudaLaunchAttributeAccessPolicyWindow, 
# 2979
cudaLaunchAttributeCooperative, 
# 2980
cudaLaunchAttributeSynchronizationPolicy, 
# 2981
cudaLaunchAttributeClusterDimension, 
# 2982
cudaLaunchAttributeClusterSchedulingPolicyPreference, 
# 2983
cudaLaunchAttributeProgrammaticStreamSerialization, 
# 2991
cudaLaunchAttributeProgrammaticEvent, 
# 3009 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
cudaLaunchAttributePriority
# 3010
} cudaLaunchAttributeID; 
#endif
# 3033 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
typedef 
# 3015
union cudaLaunchAttributeValue { 
# 3016
char pad[64]; 
# 3017
cudaAccessPolicyWindow accessPolicyWindow; 
# 3018
int cooperative; 
# 3019
cudaSynchronizationPolicy syncPolicy; 
# 3020
struct { 
# 3021
unsigned x; 
# 3022
unsigned y; 
# 3023
unsigned z; 
# 3024
} clusterDim; 
# 3025
cudaClusterSchedulingPolicy clusterSchedulingPolicyPreference; 
# 3026
int programmaticStreamSerializationAllowed; 
# 3027
struct { 
# 3028
cudaEvent_t event; 
# 3029
int flags; 
# 3030
int triggerAtBlockStart; 
# 3031
} programmaticEvent; 
# 3032
int priority; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 3033
} cudaLaunchAttributeValue; 
#endif
# 3042 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
typedef 
# 3038
struct cudaLaunchAttribute_st { 
# 3039
cudaLaunchAttributeID id; 
# 3040
char pad[(8) - sizeof(cudaLaunchAttributeID)]; 
# 3041
cudaLaunchAttributeValue val; 
# 3042
} cudaLaunchAttribute; 
#endif
# 3054 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_types.h" 3
#if 0
typedef 
# 3047
struct cudaLaunchConfig_st { 
# 3048
dim3 gridDim; 
# 3049
dim3 blockDim; 
# 3050
size_t dynamicSmemBytes; 
# 3051
cudaStream_t stream; 
# 3052
cudaLaunchAttribute *attrs; 
# 3053
unsigned numAttrs; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 3054
} cudaLaunchConfig_t; 
#endif
# 84 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_types.h" 3
#if 0
# 84
enum cudaSurfaceBoundaryMode { 
# 86
cudaBoundaryModeZero, 
# 87
cudaBoundaryModeClamp, 
# 88
cudaBoundaryModeTrap
# 89
}; 
#endif
# 94 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_types.h" 3
#if 0
# 94
enum cudaSurfaceFormatMode { 
# 96
cudaFormatModeForced, 
# 97
cudaFormatModeAuto
# 98
}; 
#endif
# 103 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_types.h" 3
#if 0
# 103
struct surfaceReference { 
# 108
cudaChannelFormatDesc channelDesc; 
# 109
}; 
#endif
# 114 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_types.h" 3
#if 0
typedef unsigned long long 
# 114
cudaSurfaceObject_t; 
#endif
# 84 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_types.h" 3
#if 0
# 84
enum cudaTextureAddressMode { 
# 86
cudaAddressModeWrap, 
# 87
cudaAddressModeClamp, 
# 88
cudaAddressModeMirror, 
# 89
cudaAddressModeBorder
# 90
}; 
#endif
# 95 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_types.h" 3
#if 0
# 95
enum cudaTextureFilterMode { 
# 97
cudaFilterModePoint, 
# 98
cudaFilterModeLinear
# 99
}; 
#endif
# 104 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_types.h" 3
#if 0
# 104
enum cudaTextureReadMode { 
# 106
cudaReadModeElementType, 
# 107
cudaReadModeNormalizedFloat
# 108
}; 
#endif
# 113 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_types.h" 3
#if 0
# 113
struct textureReference { 
# 118
int normalized; 
# 122
cudaTextureFilterMode filterMode; 
# 126
cudaTextureAddressMode addressMode[3]; 
# 130
cudaChannelFormatDesc channelDesc; 
# 134
int sRGB; 
# 138
unsigned maxAnisotropy; 
# 142
cudaTextureFilterMode mipmapFilterMode; 
# 146
float mipmapLevelBias; 
# 150
float minMipmapLevelClamp; 
# 154
float maxMipmapLevelClamp; 
# 158
int disableTrilinearOptimization; 
# 159
int __cudaReserved[14]; 
# 160
}; 
#endif
# 165 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_types.h" 3
#if 0
# 165
struct cudaTextureDesc { 
# 170
cudaTextureAddressMode addressMode[3]; 
# 174
cudaTextureFilterMode filterMode; 
# 178
cudaTextureReadMode readMode; 
# 182
int sRGB; 
# 186
float borderColor[4]; 
# 190
int normalizedCoords; 
# 194
unsigned maxAnisotropy; 
# 198
cudaTextureFilterMode mipmapFilterMode; 
# 202
float mipmapLevelBias; 
# 206
float minMipmapLevelClamp; 
# 210
float maxMipmapLevelClamp; 
# 214
int disableTrilinearOptimization; 
# 215
}; 
#endif
# 217 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_types.h" 3
#if 0
# 217
struct cudaTextureDesc_v2 { 
# 222
cudaTextureAddressMode addressMode[3]; 
# 226
cudaTextureFilterMode filterMode; 
# 230
cudaTextureReadMode readMode; 
# 234
int sRGB; 
# 238
float borderColor[4]; 
# 242
int normalizedCoords; 
# 246
unsigned maxAnisotropy; 
# 250
cudaTextureFilterMode mipmapFilterMode; 
# 254
float mipmapLevelBias; 
# 258
float minMipmapLevelClamp; 
# 262
float maxMipmapLevelClamp; 
# 266
int disableTrilinearOptimization; 
# 270
int seamlessCubemap; 
# 271
}; 
#endif
# 276 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_types.h" 3
#if 0
typedef unsigned long long 
# 276
cudaTextureObject_t; 
#endif
# 87 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/library_types.h" 3
typedef 
# 55
enum cudaDataType_t { 
# 57
CUDA_R_16F = 2, 
# 58
CUDA_C_16F = 6, 
# 59
CUDA_R_16BF = 14, 
# 60
CUDA_C_16BF, 
# 61
CUDA_R_32F = 0, 
# 62
CUDA_C_32F = 4, 
# 63
CUDA_R_64F = 1, 
# 64
CUDA_C_64F = 5, 
# 65
CUDA_R_4I = 16, 
# 66
CUDA_C_4I, 
# 67
CUDA_R_4U, 
# 68
CUDA_C_4U, 
# 69
CUDA_R_8I = 3, 
# 70
CUDA_C_8I = 7, 
# 71
CUDA_R_8U, 
# 72
CUDA_C_8U, 
# 73
CUDA_R_16I = 20, 
# 74
CUDA_C_16I, 
# 75
CUDA_R_16U, 
# 76
CUDA_C_16U, 
# 77
CUDA_R_32I = 10, 
# 78
CUDA_C_32I, 
# 79
CUDA_R_32U, 
# 80
CUDA_C_32U, 
# 81
CUDA_R_64I = 24, 
# 82
CUDA_C_64I, 
# 83
CUDA_R_64U, 
# 84
CUDA_C_64U, 
# 85
CUDA_R_8F_E4M3, 
# 86
CUDA_R_8F_E5M2
# 87
} cudaDataType; 
# 95
typedef 
# 90
enum libraryPropertyType_t { 
# 92
MAJOR_VERSION, 
# 93
MINOR_VERSION, 
# 94
PATCH_LEVEL
# 95
} libraryPropertyType; 
# 131 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_device_runtime_api.h" 3
extern "C" {
# 133
extern cudaError_t cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
# 134
extern cudaError_t cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
# 135
extern cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 136
extern cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
# 139
extern cudaError_t cudaDeviceSynchronize(); 
# 141
__attribute__((unused)) extern cudaError_t __cudaDeviceSynchronizeDeprecationAvoidance(); 
# 142
extern cudaError_t cudaGetLastError(); 
# 143
extern cudaError_t cudaPeekAtLastError(); 
# 144
extern const char *cudaGetErrorString(cudaError_t error); 
# 145
extern const char *cudaGetErrorName(cudaError_t error); 
# 146
extern cudaError_t cudaGetDeviceCount(int * count); 
# 147
extern cudaError_t cudaGetDevice(int * device); 
# 148
extern cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
# 149
extern cudaError_t cudaStreamDestroy(cudaStream_t stream); 
# 150
extern cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
# 151
__attribute__((unused)) extern cudaError_t cudaStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
# 152
extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
# 153
extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream); 
# 154
__attribute__((unused)) extern cudaError_t cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream); 
# 155
extern cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned flags); 
# 156
__attribute__((unused)) extern cudaError_t cudaEventRecordWithFlags_ptsz(cudaEvent_t event, cudaStream_t stream, unsigned flags); 
# 157
extern cudaError_t cudaEventDestroy(cudaEvent_t event); 
# 158
extern cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
# 159
extern cudaError_t cudaFree(void * devPtr); 
# 160
extern cudaError_t cudaMalloc(void ** devPtr, size_t size); 
# 161
extern cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
# 162
__attribute__((unused)) extern cudaError_t cudaMemcpyAsync_ptsz(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
# 163
extern cudaError_t cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
# 164
__attribute__((unused)) extern cudaError_t cudaMemcpy2DAsync_ptsz(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
# 165
extern cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream); 
# 166
__attribute__((unused)) extern cudaError_t cudaMemcpy3DAsync_ptsz(const cudaMemcpy3DParms * p, cudaStream_t stream); 
# 167
extern cudaError_t cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream); 
# 168
__attribute__((unused)) extern cudaError_t cudaMemsetAsync_ptsz(void * devPtr, int value, size_t count, cudaStream_t stream); 
# 169
extern cudaError_t cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
# 170
__attribute__((unused)) extern cudaError_t cudaMemset2DAsync_ptsz(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
# 171
extern cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
# 172
__attribute__((unused)) extern cudaError_t cudaMemset3DAsync_ptsz(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
# 173
extern cudaError_t cudaRuntimeGetVersion(int * runtimeVersion); 
# 194 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_device_runtime_api.h" 3
__attribute__((unused)) extern void *cudaGetParameterBuffer(size_t alignment, size_t size); 
# 222 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_device_runtime_api.h" 3
__attribute__((unused)) extern void *cudaGetParameterBufferV2(void * func, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize); 
# 223
__attribute__((unused)) extern cudaError_t cudaLaunchDevice_ptsz(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
# 224
__attribute__((unused)) extern cudaError_t cudaLaunchDeviceV2_ptsz(void * parameterBuffer, cudaStream_t stream); 
# 242 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_device_runtime_api.h" 3
__attribute__((unused)) extern cudaError_t cudaLaunchDevice(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
# 243
__attribute__((unused)) extern cudaError_t cudaLaunchDeviceV2(void * parameterBuffer, cudaStream_t stream); 
# 246
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize); 
# 247
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
# 249
__attribute__((unused)) extern unsigned long long cudaCGGetIntrinsicHandle(cudaCGScope scope); 
# 250
__attribute__((unused)) extern cudaError_t cudaCGSynchronize(unsigned long long handle, unsigned flags); 
# 251
__attribute__((unused)) extern cudaError_t cudaCGSynchronizeGrid(unsigned long long handle, unsigned flags); 
# 252
__attribute__((unused)) extern cudaError_t cudaCGGetSize(unsigned * numThreads, unsigned * numGrids, unsigned long long handle); 
# 253
__attribute__((unused)) extern cudaError_t cudaCGGetRank(unsigned * threadRank, unsigned * gridRank, unsigned long long handle); 
# 254
}
# 256
template< class T> static inline cudaError_t cudaMalloc(T ** devPtr, size_t size); 
# 257
template< class T> static inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, T * entry); 
# 258
template< class T> static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize); 
# 259
template< class T> static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
# 267 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern "C" {
# 307 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceReset(); 
# 329 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceSynchronize(); 
# 416 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value); 
# 449 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
# 472 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t * maxWidthInElements, const cudaChannelFormatDesc * fmtDesc, int device); 
# 506 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 543 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority); 
# 587 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig); 
# 618 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
# 662 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config); 
# 689 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetByPCIBusId(int * device, const char * pciBusId); 
# 719 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetPCIBusId(char * pciBusId, int len, int device); 
# 767 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t event); 
# 808 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t handle); 
# 851 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr); 
# 915 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t handle, unsigned flags); 
# 951 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaIpcCloseMemHandle(void * devPtr); 
# 983 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope); 
# 1026 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaThreadExit(); 
# 1052 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaThreadSynchronize(); 
# 1101 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaThreadSetLimit(cudaLimit limit, size_t value); 
# 1134 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaThreadGetLimit(size_t * pValue, cudaLimit limit); 
# 1170 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaThreadGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 1217 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaThreadSetCacheConfig(cudaFuncCache cacheConfig); 
# 1278 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetLastError(); 
# 1326 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaPeekAtLastError(); 
# 1342 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern const char *cudaGetErrorName(cudaError_t error); 
# 1358 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern const char *cudaGetErrorString(cudaError_t error); 
# 1386 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetDeviceCount(int * count); 
# 1659 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetDeviceProperties(cudaDeviceProp * prop, int device); 
# 1859 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
# 1877 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t * memPool, int device); 
# 1901 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool); 
# 1921 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetMemPool(cudaMemPool_t * memPool, int device); 
# 1969 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetNvSciSyncAttributes(void * nvSciSyncAttrList, int device, int flags); 
# 2009 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetP2PAttribute(int * value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice); 
# 2030 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaChooseDevice(int * device, const cudaDeviceProp * prop); 
# 2074 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaSetDevice(int device); 
# 2095 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetDevice(int * device); 
# 2126 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaSetValidDevices(int * device_arr, int len); 
# 2191 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaSetDeviceFlags(unsigned flags); 
# 2235 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetDeviceFlags(unsigned * flags); 
# 2275 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamCreate(cudaStream_t * pStream); 
# 2307 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
# 2353 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamCreateWithPriority(cudaStream_t * pStream, unsigned flags, int priority); 
# 2380 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int * priority); 
# 2405 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned * flags); 
# 2420 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaCtxResetPersistingL2Cache(); 
# 2440 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src); 
# 2461 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamGetAttribute(cudaStream_t hStream, cudaLaunchAttributeID attr, cudaLaunchAttributeValue * value_out); 
# 2485 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamSetAttribute(cudaStream_t hStream, cudaLaunchAttributeID attr, const cudaLaunchAttributeValue * value); 
# 2519 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamDestroy(cudaStream_t stream); 
# 2550 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags = 0); 
# 2558
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void * userData); 
# 2625 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void * userData, unsigned flags); 
# 2649 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamSynchronize(cudaStream_t stream); 
# 2674 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamQuery(cudaStream_t stream); 
# 2758 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void * devPtr, size_t length = 0, unsigned flags = 4); 
# 2797 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode); 
# 2848 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode * mode); 
# 2876 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t * pGraph); 
# 2914 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus * pCaptureStatus); 
# 2946 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus * pCaptureStatus, unsigned long long * pId); 
# 3001 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, cudaStreamCaptureStatus * captureStatus_out, unsigned long long * id_out = 0, cudaGraph_t * graph_out = 0, const cudaGraphNode_t ** dependencies_out = 0, size_t * numDependencies_out = 0); 
# 3034 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t * dependencies, size_t numDependencies, unsigned flags = 0); 
# 3071 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaEventCreate(cudaEvent_t * event); 
# 3108 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
# 3148 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0); 
# 3195 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream = 0, unsigned flags = 0); 
# 3227 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaEventQuery(cudaEvent_t event); 
# 3257 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaEventSynchronize(cudaEvent_t event); 
# 3286 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaEventDestroy(cudaEvent_t event); 
# 3330 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end); 
# 3510 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaImportExternalMemory(cudaExternalMemory_t * extMem_out, const cudaExternalMemoryHandleDesc * memHandleDesc); 
# 3565 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaExternalMemoryGetMappedBuffer(void ** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc * bufferDesc); 
# 3627 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t * mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc * mipmapDesc); 
# 3651 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem); 
# 3804 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t * extSem_out, const cudaExternalSemaphoreHandleDesc * semHandleDesc); 
# 3871 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaSignalExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t * extSemArray, const cudaExternalSemaphoreSignalParams * paramsArray, unsigned numExtSems, cudaStream_t stream = 0); 
# 3947 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaWaitExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t * extSemArray, const cudaExternalSemaphoreWaitParams * paramsArray, unsigned numExtSems, cudaStream_t stream = 0); 
# 3970 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem); 
# 4037 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaLaunchKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
# 4099 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t * config, const void * func, void ** args); 
# 4156 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaLaunchCooperativeKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
# 4257 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaLaunchCooperativeKernelMultiDevice(cudaLaunchParams * launchParamsList, unsigned numDevices, unsigned flags = 0); 
# 4304 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaFuncSetCacheConfig(const void * func, cudaFuncCache cacheConfig); 
# 4359 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaFuncSetSharedMemConfig(const void * func, cudaSharedMemConfig config); 
# 4392 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
# 4429 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaFuncSetAttribute(const void * func, cudaFuncAttribute attr, int value); 
# 4453 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaSetDoubleForDevice(double * d); 
# 4477 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaSetDoubleForHost(double * d); 
# 4543 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void * userData); 
# 4600 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize); 
# 4629 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t * dynamicSmemSize, const void * func, int numBlocks, int blockSize); 
# 4674 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize, unsigned flags); 
# 4709 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaOccupancyMaxPotentialClusterSize(int * clusterSize, const void * func, const cudaLaunchConfig_t * launchConfig); 
# 4748 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaOccupancyMaxActiveClusters(int * numClusters, const void * func, const cudaLaunchConfig_t * launchConfig); 
# 4868 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMallocManaged(void ** devPtr, size_t size, unsigned flags = 1); 
# 4901 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMalloc(void ** devPtr, size_t size); 
# 4934 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMallocHost(void ** ptr, size_t size); 
# 4977 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height); 
# 5029 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMallocArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, size_t width, size_t height = 0, unsigned flags = 0); 
# 5067 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaFree(void * devPtr); 
# 5090 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaFreeHost(void * ptr); 
# 5113 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaFreeArray(cudaArray_t array); 
# 5136 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray); 
# 5202 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaHostAlloc(void ** pHost, size_t size, unsigned flags); 
# 5295 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaHostRegister(void * ptr, size_t size, unsigned flags); 
# 5318 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaHostUnregister(void * ptr); 
# 5363 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned flags); 
# 5385 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaHostGetFlags(unsigned * pFlags, void * pHost); 
# 5424 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMalloc3D(cudaPitchedPtr * pitchedDevPtr, cudaExtent extent); 
# 5569 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMalloc3DArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned flags = 0); 
# 5714 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t * mipmappedArray, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned numLevels, unsigned flags = 0); 
# 5747 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t * levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned level); 
# 5852 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms * p); 
# 5883 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms * p); 
# 6001 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream = 0); 
# 6027 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms * p, cudaStream_t stream = 0); 
# 6061 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemGetInfo(size_t * free, size_t * total); 
# 6087 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc * desc, cudaExtent * extent, unsigned * flags, cudaArray_t array); 
# 6116 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaArrayGetPlane(cudaArray_t * pPlaneArray, cudaArray_t hArray, unsigned planeIdx); 
# 6139 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaArrayGetMemoryRequirements(cudaArrayMemoryRequirements * memoryRequirements, cudaArray_t array, int device); 
# 6163 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMipmappedArrayGetMemoryRequirements(cudaArrayMemoryRequirements * memoryRequirements, cudaMipmappedArray_t mipmap, int device); 
# 6191 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaArrayGetSparseProperties(cudaArraySparseProperties * sparseProperties, cudaArray_t array); 
# 6221 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties * sparseProperties, cudaMipmappedArray_t mipmap); 
# 6266 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy(void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 6301 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpyPeer(void * dst, int dstDevice, const void * src, int srcDevice, size_t count); 
# 6350 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
# 6400 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
# 6450 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy2DFromArray(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind); 
# 6497 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
# 6540 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpyToSymbol(const void * symbol, const void * src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice); 
# 6583 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpyFromSymbol(void * dst, const void * symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost); 
# 6640 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6675 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpyPeerAsync(void * dst, int dstDevice, const void * src, int srcDevice, size_t count, cudaStream_t stream = 0); 
# 6738 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6796 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6853 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy2DFromArrayAsync(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6904 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpyToSymbolAsync(const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6955 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6984 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemset(void * devPtr, int value, size_t count); 
# 7018 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemset2D(void * devPtr, size_t pitch, int value, size_t width, size_t height); 
# 7064 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent); 
# 7100 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream = 0); 
# 7141 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0); 
# 7194 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream = 0); 
# 7222 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetSymbolAddress(void ** devPtr, const void * symbol); 
# 7249 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetSymbolSize(size_t * size, const void * symbol); 
# 7319 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPrefetchAsync(const void * devPtr, size_t count, int dstDevice, cudaStream_t stream = 0); 
# 7435 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemAdvise(const void * devPtr, size_t count, cudaMemoryAdvise advice, int device); 
# 7494 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemRangeGetAttribute(void * data, size_t dataSize, cudaMemRangeAttribute attribute, const void * devPtr, size_t count); 
# 7533 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemRangeGetAttributes(void ** data, size_t * dataSizes, cudaMemRangeAttribute * attributes, size_t numAttributes, const void * devPtr, size_t count); 
# 7593 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind); 
# 7635 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaMemcpyFromArray(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind); 
# 7678 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
# 7729 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 7779 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaMemcpyFromArrayAsync(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 7848 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMallocAsync(void ** devPtr, size_t size, cudaStream_t hStream); 
# 7874 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaFreeAsync(void * devPtr, cudaStream_t hStream); 
# 7899 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep); 
# 7943 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void * value); 
# 7991 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void * value); 
# 8006 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc * descList, size_t count); 
# 8019 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPoolGetAccess(cudaMemAccessFlags * flags, cudaMemPool_t memPool, cudaMemLocation * location); 
# 8039 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPoolCreate(cudaMemPool_t * memPool, const cudaMemPoolProps * poolProps); 
# 8061 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool); 
# 8097 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMallocFromPoolAsync(void ** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream); 
# 8122 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPoolExportToShareableHandle(void * shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned flags); 
# 8149 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t * memPool, void * shareableHandle, cudaMemAllocationHandleType handleType, unsigned flags); 
# 8172 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPoolExportPointer(cudaMemPoolPtrExportData * exportData, void * ptr); 
# 8201 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemPoolImportPointer(void ** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData * exportData); 
# 8353 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaPointerGetAttributes(cudaPointerAttributes * attributes, const void * ptr); 
# 8394 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceCanAccessPeer(int * canAccessPeer, int device, int peerDevice); 
# 8436 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned flags); 
# 8458 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceDisablePeerAccess(int peerDevice); 
# 8522 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource); 
# 8557 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned flags); 
# 8596 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
# 8631 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
# 8663 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, cudaGraphicsResource_t resource); 
# 8701 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t * array, cudaGraphicsResource_t resource, unsigned arrayIndex, unsigned mipLevel); 
# 8730 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t resource); 
# 8801 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaBindTexture(size_t * offset, const textureReference * texref, const void * devPtr, const cudaChannelFormatDesc * desc, size_t size = ((2147483647) * 2U) + 1U); 
# 8860 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaBindTexture2D(size_t * offset, const textureReference * texref, const void * devPtr, const cudaChannelFormatDesc * desc, size_t width, size_t height, size_t pitch); 
# 8898 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaBindTextureToArray(const textureReference * texref, cudaArray_const_t array, const cudaChannelFormatDesc * desc); 
# 8938 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaBindTextureToMipmappedArray(const textureReference * texref, cudaMipmappedArray_const_t mipmappedArray, const cudaChannelFormatDesc * desc); 
# 8964 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaUnbindTexture(const textureReference * texref); 
# 8993 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaGetTextureAlignmentOffset(size_t * offset, const textureReference * texref); 
# 9023 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaGetTextureReference(const textureReference ** texref, const void * symbol); 
# 9068 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaBindSurfaceToArray(const surfaceReference * surfref, cudaArray_const_t array, const cudaChannelFormatDesc * desc); 
# 9093 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
__attribute((deprecated)) extern cudaError_t cudaGetSurfaceReference(const surfaceReference ** surfref, const void * symbol); 
# 9128 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc * desc, cudaArray_const_t array); 
# 9158 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f); 
# 9375 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const cudaResourceDesc * pResDesc, const cudaTextureDesc * pTexDesc, const cudaResourceViewDesc * pResViewDesc); 
# 9599 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaCreateTextureObject_v2(cudaTextureObject_t * pTexObject, const cudaResourceDesc * pResDesc, const cudaTextureDesc_v2 * pTexDesc, const cudaResourceViewDesc * pResViewDesc); 
# 9619 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject); 
# 9639 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc * pResDesc, cudaTextureObject_t texObject); 
# 9659 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc * pTexDesc, cudaTextureObject_t texObject); 
# 9679 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetTextureObjectTextureDesc_v2(cudaTextureDesc_v2 * pTexDesc, cudaTextureObject_t texObject); 
# 9700 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t texObject); 
# 9745 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t * pSurfObject, const cudaResourceDesc * pResDesc); 
# 9765 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject); 
# 9784 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc * pResDesc, cudaSurfaceObject_t surfObject); 
# 9818 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDriverGetVersion(int * driverVersion); 
# 9843 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaRuntimeGetVersion(int * runtimeVersion); 
# 9890 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphCreate(cudaGraph_t * pGraph, unsigned flags); 
# 9987 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaKernelNodeParams * pNodeParams); 
# 10020 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams * pNodeParams); 
# 10045 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams); 
# 10065 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst); 
# 10088 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaLaunchAttributeID attr, cudaLaunchAttributeValue * value_out); 
# 10112 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaLaunchAttributeID attr, const cudaLaunchAttributeValue * value); 
# 10162 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemcpy3DParms * pCopyParams); 
# 10221 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
# 10290 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
# 10358 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 10390 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms * pNodeParams); 
# 10416 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms * pNodeParams); 
# 10455 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
# 10501 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
# 10547 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 10594 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemsetParams * pMemsetParams); 
# 10617 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams * pNodeParams); 
# 10640 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams * pNodeParams); 
# 10681 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddHostNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaHostNodeParams * pNodeParams); 
# 10704 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams * pNodeParams); 
# 10727 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams * pNodeParams); 
# 10767 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaGraph_t childGraph); 
# 10794 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t * pGraph); 
# 10831 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies); 
# 10874 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaEvent_t event); 
# 10901 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t * event_out); 
# 10928 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event); 
# 10974 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaEvent_t event); 
# 11001 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t * event_out); 
# 11028 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event); 
# 11077 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
# 11110 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams * params_out); 
# 11137 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
# 11186 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
# 11219 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams * params_out); 
# 11246 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
# 11323 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaMemAllocNodeParams * nodeParams); 
# 11350 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams * params_out); 
# 11410 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dptr); 
# 11434 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void * dptr_out); 
# 11462 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGraphMemTrim(int device); 
# 11499 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void * value); 
# 11533 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void * value); 
# 11561 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphClone(cudaGraph_t * pGraphClone, cudaGraph_t originalGraph); 
# 11589 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t * pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph); 
# 11620 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType * pType); 
# 11651 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t * nodes, size_t * numNodes); 
# 11682 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t * pRootNodes, size_t * pNumRootNodes); 
# 11716 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t * from, cudaGraphNode_t * to, size_t * numEdges); 
# 11747 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t * pDependencies, size_t * pNumDependencies); 
# 11779 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t * pDependentNodes, size_t * pNumDependentNodes); 
# 11810 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t numDependencies); 
# 11841 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t numDependencies); 
# 11871 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node); 
# 11909 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphInstantiate(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, cudaGraphNode_t * pErrorNode, char * pLogBuffer, size_t bufferSize); 
# 11957 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, unsigned long long flags); 
# 12001 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams); 
# 12051 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms * pNodeParams); 
# 12106 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
# 12169 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
# 12230 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 12284 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams * pNodeParams); 
# 12323 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams * pNodeParams); 
# 12369 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph); 
# 12413 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event); 
# 12457 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event); 
# 12504 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
# 12551 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
# 12591 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned isEnabled); 
# 12625 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned * isEnabled); 
# 12706 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t * hErrorNode_out, cudaGraphExecUpdateResult * updateResult_out); 
# 12731 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream); 
# 12762 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream); 
# 12785 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec); 
# 12806 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphDestroy(cudaGraph_t graph); 
# 12825 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char * path, unsigned flags); 
# 12861 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaUserObjectCreate(cudaUserObject_t * object_out, void * ptr, cudaHostFn_t destroy, unsigned initialRefcount, unsigned flags); 
# 12885 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned count = 1); 
# 12913 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned count = 1); 
# 12941 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned count = 1, unsigned flags = 0); 
# 12966 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned count = 1); 
# 13032 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetDriverEntryPoint(const char * symbol, void ** funcPtr, unsigned long long flags); 
# 13037
extern cudaError_t cudaGetExportTable(const void ** ppExportTable, const cudaUUID_t * pExportTableId); 
# 13213 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
extern cudaError_t cudaGetFuncBySymbol(cudaFunction_t * functionPtr, const void * symbolPtr); 
# 13365 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime_api.h" 3
}
# 124 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/channel_descriptor.h" 3
template< class T> inline cudaChannelFormatDesc cudaCreateChannelDesc() 
# 125
{ 
# 126
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone); 
# 127
} 
# 129
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf() 
# 130
{ 
# 131
int e = (((int)sizeof(unsigned short)) * 8); 
# 133
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 134
} 
# 136
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf1() 
# 137
{ 
# 138
int e = (((int)sizeof(unsigned short)) * 8); 
# 140
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 141
} 
# 143
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf2() 
# 144
{ 
# 145
int e = (((int)sizeof(unsigned short)) * 8); 
# 147
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
# 148
} 
# 150
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf4() 
# 151
{ 
# 152
int e = (((int)sizeof(unsigned short)) * 8); 
# 154
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
# 155
} 
# 157
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char> () 
# 158
{ 
# 159
int e = (((int)sizeof(char)) * 8); 
# 164
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 166
} 
# 168
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< signed char> () 
# 169
{ 
# 170
int e = (((int)sizeof(signed char)) * 8); 
# 172
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 173
} 
# 175
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned char> () 
# 176
{ 
# 177
int e = (((int)sizeof(unsigned char)) * 8); 
# 179
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 180
} 
# 182
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char1> () 
# 183
{ 
# 184
int e = (((int)sizeof(signed char)) * 8); 
# 186
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 187
} 
# 189
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar1> () 
# 190
{ 
# 191
int e = (((int)sizeof(unsigned char)) * 8); 
# 193
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 194
} 
# 196
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char2> () 
# 197
{ 
# 198
int e = (((int)sizeof(signed char)) * 8); 
# 200
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 201
} 
# 203
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar2> () 
# 204
{ 
# 205
int e = (((int)sizeof(unsigned char)) * 8); 
# 207
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 208
} 
# 210
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char4> () 
# 211
{ 
# 212
int e = (((int)sizeof(signed char)) * 8); 
# 214
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 215
} 
# 217
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar4> () 
# 218
{ 
# 219
int e = (((int)sizeof(unsigned char)) * 8); 
# 221
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 222
} 
# 224
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short> () 
# 225
{ 
# 226
int e = (((int)sizeof(short)) * 8); 
# 228
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 229
} 
# 231
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned short> () 
# 232
{ 
# 233
int e = (((int)sizeof(unsigned short)) * 8); 
# 235
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 236
} 
# 238
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short1> () 
# 239
{ 
# 240
int e = (((int)sizeof(short)) * 8); 
# 242
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 243
} 
# 245
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort1> () 
# 246
{ 
# 247
int e = (((int)sizeof(unsigned short)) * 8); 
# 249
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 250
} 
# 252
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short2> () 
# 253
{ 
# 254
int e = (((int)sizeof(short)) * 8); 
# 256
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 257
} 
# 259
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort2> () 
# 260
{ 
# 261
int e = (((int)sizeof(unsigned short)) * 8); 
# 263
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 264
} 
# 266
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short4> () 
# 267
{ 
# 268
int e = (((int)sizeof(short)) * 8); 
# 270
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 271
} 
# 273
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort4> () 
# 274
{ 
# 275
int e = (((int)sizeof(unsigned short)) * 8); 
# 277
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 278
} 
# 280
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int> () 
# 281
{ 
# 282
int e = (((int)sizeof(int)) * 8); 
# 284
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 285
} 
# 287
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned> () 
# 288
{ 
# 289
int e = (((int)sizeof(unsigned)) * 8); 
# 291
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 292
} 
# 294
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int1> () 
# 295
{ 
# 296
int e = (((int)sizeof(int)) * 8); 
# 298
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 299
} 
# 301
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint1> () 
# 302
{ 
# 303
int e = (((int)sizeof(unsigned)) * 8); 
# 305
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 306
} 
# 308
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int2> () 
# 309
{ 
# 310
int e = (((int)sizeof(int)) * 8); 
# 312
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 313
} 
# 315
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint2> () 
# 316
{ 
# 317
int e = (((int)sizeof(unsigned)) * 8); 
# 319
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 320
} 
# 322
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int4> () 
# 323
{ 
# 324
int e = (((int)sizeof(int)) * 8); 
# 326
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 327
} 
# 329
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint4> () 
# 330
{ 
# 331
int e = (((int)sizeof(unsigned)) * 8); 
# 333
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 334
} 
# 396 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/channel_descriptor.h" 3
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float> () 
# 397
{ 
# 398
int e = (((int)sizeof(float)) * 8); 
# 400
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 401
} 
# 403
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float1> () 
# 404
{ 
# 405
int e = (((int)sizeof(float)) * 8); 
# 407
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 408
} 
# 410
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float2> () 
# 411
{ 
# 412
int e = (((int)sizeof(float)) * 8); 
# 414
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
# 415
} 
# 417
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float4> () 
# 418
{ 
# 419
int e = (((int)sizeof(float)) * 8); 
# 421
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
# 422
} 
# 424
static inline cudaChannelFormatDesc cudaCreateChannelDescNV12() 
# 425
{ 
# 426
int e = (((int)sizeof(char)) * 8); 
# 428
return cudaCreateChannelDesc(e, e, e, 0, cudaChannelFormatKindNV12); 
# 429
} 
# 431
template< cudaChannelFormatKind > inline cudaChannelFormatDesc cudaCreateChannelDesc() 
# 432
{ 
# 433
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone); 
# 434
} 
# 437
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized8X1> () 
# 438
{ 
# 439
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindSignedNormalized8X1); 
# 440
} 
# 442
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized8X2> () 
# 443
{ 
# 444
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindSignedNormalized8X2); 
# 445
} 
# 447
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized8X4> () 
# 448
{ 
# 449
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindSignedNormalized8X4); 
# 450
} 
# 453
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized8X1> () 
# 454
{ 
# 455
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsignedNormalized8X1); 
# 456
} 
# 458
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized8X2> () 
# 459
{ 
# 460
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsignedNormalized8X2); 
# 461
} 
# 463
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized8X4> () 
# 464
{ 
# 465
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedNormalized8X4); 
# 466
} 
# 469
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized16X1> () 
# 470
{ 
# 471
return cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindSignedNormalized16X1); 
# 472
} 
# 474
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized16X2> () 
# 475
{ 
# 476
return cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindSignedNormalized16X2); 
# 477
} 
# 479
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized16X4> () 
# 480
{ 
# 481
return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindSignedNormalized16X4); 
# 482
} 
# 485
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized16X1> () 
# 486
{ 
# 487
return cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsignedNormalized16X1); 
# 488
} 
# 490
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized16X2> () 
# 491
{ 
# 492
return cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindUnsignedNormalized16X2); 
# 493
} 
# 495
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized16X4> () 
# 496
{ 
# 497
return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsignedNormalized16X4); 
# 498
} 
# 501
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindNV12> () 
# 502
{ 
# 503
return cudaCreateChannelDesc(8, 8, 8, 0, cudaChannelFormatKindNV12); 
# 504
} 
# 507
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed1> () 
# 508
{ 
# 509
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed1); 
# 510
} 
# 513
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed1SRGB> () 
# 514
{ 
# 515
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed1SRGB); 
# 516
} 
# 519
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed2> () 
# 520
{ 
# 521
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed2); 
# 522
} 
# 525
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed2SRGB> () 
# 526
{ 
# 527
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed2SRGB); 
# 528
} 
# 531
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed3> () 
# 532
{ 
# 533
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed3); 
# 534
} 
# 537
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed3SRGB> () 
# 538
{ 
# 539
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed3SRGB); 
# 540
} 
# 543
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed4> () 
# 544
{ 
# 545
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsignedBlockCompressed4); 
# 546
} 
# 549
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedBlockCompressed4> () 
# 550
{ 
# 551
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindSignedBlockCompressed4); 
# 552
} 
# 555
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed5> () 
# 556
{ 
# 557
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsignedBlockCompressed5); 
# 558
} 
# 561
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedBlockCompressed5> () 
# 562
{ 
# 563
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindSignedBlockCompressed5); 
# 564
} 
# 567
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed6H> () 
# 568
{ 
# 569
return cudaCreateChannelDesc(16, 16, 16, 0, cudaChannelFormatKindUnsignedBlockCompressed6H); 
# 570
} 
# 573
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedBlockCompressed6H> () 
# 574
{ 
# 575
return cudaCreateChannelDesc(16, 16, 16, 0, cudaChannelFormatKindSignedBlockCompressed6H); 
# 576
} 
# 579
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed7> () 
# 580
{ 
# 581
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed7); 
# 582
} 
# 585
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed7SRGB> () 
# 586
{ 
# 587
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed7SRGB); 
# 588
} 
# 79 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_functions.h" 3
static inline cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz) 
# 80
{ 
# 81
cudaPitchedPtr s; 
# 83
(s.ptr) = d; 
# 84
(s.pitch) = p; 
# 85
(s.xsize) = xsz; 
# 86
(s.ysize) = ysz; 
# 88
return s; 
# 89
} 
# 106 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_functions.h" 3
static inline cudaPos make_cudaPos(size_t x, size_t y, size_t z) 
# 107
{ 
# 108
cudaPos p; 
# 110
(p.x) = x; 
# 111
(p.y) = y; 
# 112
(p.z) = z; 
# 114
return p; 
# 115
} 
# 132 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/driver_functions.h" 3
static inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) 
# 133
{ 
# 134
cudaExtent e; 
# 136
(e.width) = w; 
# 137
(e.height) = h; 
# 138
(e.depth) = d; 
# 140
return e; 
# 141
} 
# 73 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_functions.h" 3
static inline char1 make_char1(signed char x); 
# 75
static inline uchar1 make_uchar1(unsigned char x); 
# 77
static inline char2 make_char2(signed char x, signed char y); 
# 79
static inline uchar2 make_uchar2(unsigned char x, unsigned char y); 
# 81
static inline char3 make_char3(signed char x, signed char y, signed char z); 
# 83
static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z); 
# 85
static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w); 
# 87
static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w); 
# 89
static inline short1 make_short1(short x); 
# 91
static inline ushort1 make_ushort1(unsigned short x); 
# 93
static inline short2 make_short2(short x, short y); 
# 95
static inline ushort2 make_ushort2(unsigned short x, unsigned short y); 
# 97
static inline short3 make_short3(short x, short y, short z); 
# 99
static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z); 
# 101
static inline short4 make_short4(short x, short y, short z, short w); 
# 103
static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w); 
# 105
static inline int1 make_int1(int x); 
# 107
static inline uint1 make_uint1(unsigned x); 
# 109
static inline int2 make_int2(int x, int y); 
# 111
static inline uint2 make_uint2(unsigned x, unsigned y); 
# 113
static inline int3 make_int3(int x, int y, int z); 
# 115
static inline uint3 make_uint3(unsigned x, unsigned y, unsigned z); 
# 117
static inline int4 make_int4(int x, int y, int z, int w); 
# 119
static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w); 
# 121
static inline long1 make_long1(long x); 
# 123
static inline ulong1 make_ulong1(unsigned long x); 
# 125
static inline long2 make_long2(long x, long y); 
# 127
static inline ulong2 make_ulong2(unsigned long x, unsigned long y); 
# 129
static inline long3 make_long3(long x, long y, long z); 
# 131
static inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z); 
# 133
static inline long4 make_long4(long x, long y, long z, long w); 
# 135
static inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w); 
# 137
static inline float1 make_float1(float x); 
# 139
static inline float2 make_float2(float x, float y); 
# 141
static inline float3 make_float3(float x, float y, float z); 
# 143
static inline float4 make_float4(float x, float y, float z, float w); 
# 145
static inline longlong1 make_longlong1(long long x); 
# 147
static inline ulonglong1 make_ulonglong1(unsigned long long x); 
# 149
static inline longlong2 make_longlong2(long long x, long long y); 
# 151
static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y); 
# 153
static inline longlong3 make_longlong3(long long x, long long y, long long z); 
# 155
static inline ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z); 
# 157
static inline longlong4 make_longlong4(long long x, long long y, long long z, long long w); 
# 159
static inline ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w); 
# 161
static inline double1 make_double1(double x); 
# 163
static inline double2 make_double2(double x, double y); 
# 165
static inline double3 make_double3(double x, double y, double z); 
# 167
static inline double4 make_double4(double x, double y, double z, double w); 
# 73 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/vector_functions.hpp" 3
static inline char1 make_char1(signed char x) 
# 74
{ 
# 75
char1 t; (t.x) = x; return t; 
# 76
} 
# 78
static inline uchar1 make_uchar1(unsigned char x) 
# 79
{ 
# 80
uchar1 t; (t.x) = x; return t; 
# 81
} 
# 83
static inline char2 make_char2(signed char x, signed char y) 
# 84
{ 
# 85
char2 t; (t.x) = x; (t.y) = y; return t; 
# 86
} 
# 88
static inline uchar2 make_uchar2(unsigned char x, unsigned char y) 
# 89
{ 
# 90
uchar2 t; (t.x) = x; (t.y) = y; return t; 
# 91
} 
# 93
static inline char3 make_char3(signed char x, signed char y, signed char z) 
# 94
{ 
# 95
char3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 96
} 
# 98
static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z) 
# 99
{ 
# 100
uchar3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 101
} 
# 103
static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w) 
# 104
{ 
# 105
char4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 106
} 
# 108
static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) 
# 109
{ 
# 110
uchar4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 111
} 
# 113
static inline short1 make_short1(short x) 
# 114
{ 
# 115
short1 t; (t.x) = x; return t; 
# 116
} 
# 118
static inline ushort1 make_ushort1(unsigned short x) 
# 119
{ 
# 120
ushort1 t; (t.x) = x; return t; 
# 121
} 
# 123
static inline short2 make_short2(short x, short y) 
# 124
{ 
# 125
short2 t; (t.x) = x; (t.y) = y; return t; 
# 126
} 
# 128
static inline ushort2 make_ushort2(unsigned short x, unsigned short y) 
# 129
{ 
# 130
ushort2 t; (t.x) = x; (t.y) = y; return t; 
# 131
} 
# 133
static inline short3 make_short3(short x, short y, short z) 
# 134
{ 
# 135
short3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 136
} 
# 138
static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z) 
# 139
{ 
# 140
ushort3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 141
} 
# 143
static inline short4 make_short4(short x, short y, short z, short w) 
# 144
{ 
# 145
short4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 146
} 
# 148
static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w) 
# 149
{ 
# 150
ushort4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 151
} 
# 153
static inline int1 make_int1(int x) 
# 154
{ 
# 155
int1 t; (t.x) = x; return t; 
# 156
} 
# 158
static inline uint1 make_uint1(unsigned x) 
# 159
{ 
# 160
uint1 t; (t.x) = x; return t; 
# 161
} 
# 163
static inline int2 make_int2(int x, int y) 
# 164
{ 
# 165
int2 t; (t.x) = x; (t.y) = y; return t; 
# 166
} 
# 168
static inline uint2 make_uint2(unsigned x, unsigned y) 
# 169
{ 
# 170
uint2 t; (t.x) = x; (t.y) = y; return t; 
# 171
} 
# 173
static inline int3 make_int3(int x, int y, int z) 
# 174
{ 
# 175
int3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 176
} 
# 178
static inline uint3 make_uint3(unsigned x, unsigned y, unsigned z) 
# 179
{ 
# 180
uint3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 181
} 
# 183
static inline int4 make_int4(int x, int y, int z, int w) 
# 184
{ 
# 185
int4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 186
} 
# 188
static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) 
# 189
{ 
# 190
uint4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 191
} 
# 193
static inline long1 make_long1(long x) 
# 194
{ 
# 195
long1 t; (t.x) = x; return t; 
# 196
} 
# 198
static inline ulong1 make_ulong1(unsigned long x) 
# 199
{ 
# 200
ulong1 t; (t.x) = x; return t; 
# 201
} 
# 203
static inline long2 make_long2(long x, long y) 
# 204
{ 
# 205
long2 t; (t.x) = x; (t.y) = y; return t; 
# 206
} 
# 208
static inline ulong2 make_ulong2(unsigned long x, unsigned long y) 
# 209
{ 
# 210
ulong2 t; (t.x) = x; (t.y) = y; return t; 
# 211
} 
# 213
static inline long3 make_long3(long x, long y, long z) 
# 214
{ 
# 215
long3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 216
} 
# 218
static inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z) 
# 219
{ 
# 220
ulong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 221
} 
# 223
static inline long4 make_long4(long x, long y, long z, long w) 
# 224
{ 
# 225
long4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 226
} 
# 228
static inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w) 
# 229
{ 
# 230
ulong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 231
} 
# 233
static inline float1 make_float1(float x) 
# 234
{ 
# 235
float1 t; (t.x) = x; return t; 
# 236
} 
# 238
static inline float2 make_float2(float x, float y) 
# 239
{ 
# 240
float2 t; (t.x) = x; (t.y) = y; return t; 
# 241
} 
# 243
static inline float3 make_float3(float x, float y, float z) 
# 244
{ 
# 245
float3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 246
} 
# 248
static inline float4 make_float4(float x, float y, float z, float w) 
# 249
{ 
# 250
float4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 251
} 
# 253
static inline longlong1 make_longlong1(long long x) 
# 254
{ 
# 255
longlong1 t; (t.x) = x; return t; 
# 256
} 
# 258
static inline ulonglong1 make_ulonglong1(unsigned long long x) 
# 259
{ 
# 260
ulonglong1 t; (t.x) = x; return t; 
# 261
} 
# 263
static inline longlong2 make_longlong2(long long x, long long y) 
# 264
{ 
# 265
longlong2 t; (t.x) = x; (t.y) = y; return t; 
# 266
} 
# 268
static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y) 
# 269
{ 
# 270
ulonglong2 t; (t.x) = x; (t.y) = y; return t; 
# 271
} 
# 273
static inline longlong3 make_longlong3(long long x, long long y, long long z) 
# 274
{ 
# 275
longlong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 276
} 
# 278
static inline ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z) 
# 279
{ 
# 280
ulonglong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 281
} 
# 283
static inline longlong4 make_longlong4(long long x, long long y, long long z, long long w) 
# 284
{ 
# 285
longlong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 286
} 
# 288
static inline ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w) 
# 289
{ 
# 290
ulonglong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 291
} 
# 293
static inline double1 make_double1(double x) 
# 294
{ 
# 295
double1 t; (t.x) = x; return t; 
# 296
} 
# 298
static inline double2 make_double2(double x, double y) 
# 299
{ 
# 300
double2 t; (t.x) = x; (t.y) = y; return t; 
# 301
} 
# 303
static inline double3 make_double3(double x, double y, double z) 
# 304
{ 
# 305
double3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 306
} 
# 308
static inline double4 make_double4(double x, double y, double z, double w) 
# 309
{ 
# 310
double4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 311
} 
# 28 "/usr/include/string.h" 3
extern "C" {
# 43 "/usr/include/string.h" 3
extern void *memcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 44
 __attribute((__nonnull__(1, 2))); 
# 47
extern void *memmove(void * __dest, const void * __src, size_t __n) throw()
# 48
 __attribute((__nonnull__(1, 2))); 
# 54
extern void *memccpy(void *__restrict__ __dest, const void *__restrict__ __src, int __c, size_t __n) throw()
# 56
 __attribute((__nonnull__(1, 2))); 
# 61
extern void *memset(void * __s, int __c, size_t __n) throw() __attribute((__nonnull__(1))); 
# 64
extern int memcmp(const void * __s1, const void * __s2, size_t __n) throw()
# 65
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 69
extern "C++" {
# 71
extern void *memchr(void * __s, int __c, size_t __n) throw() __asm__("memchr")
# 72
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 73
extern const void *memchr(const void * __s, int __c, size_t __n) throw() __asm__("memchr")
# 74
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 89 "/usr/include/string.h" 3
}
# 99 "/usr/include/string.h" 3
extern "C++" void *rawmemchr(void * __s, int __c) throw() __asm__("rawmemchr")
# 100
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 101
extern "C++" const void *rawmemchr(const void * __s, int __c) throw() __asm__("rawmemchr")
# 102
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 110
extern "C++" void *memrchr(void * __s, int __c, size_t __n) throw() __asm__("memrchr")
# 111
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 112
extern "C++" const void *memrchr(const void * __s, int __c, size_t __n) throw() __asm__("memrchr")
# 113
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 122 "/usr/include/string.h" 3
extern char *strcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 123
 __attribute((__nonnull__(1, 2))); 
# 125
extern char *strncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 127
 __attribute((__nonnull__(1, 2))); 
# 130
extern char *strcat(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 131
 __attribute((__nonnull__(1, 2))); 
# 133
extern char *strncat(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 134
 __attribute((__nonnull__(1, 2))); 
# 137
extern int strcmp(const char * __s1, const char * __s2) throw()
# 138
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 140
extern int strncmp(const char * __s1, const char * __s2, size_t __n) throw()
# 141
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 144
extern int strcoll(const char * __s1, const char * __s2) throw()
# 145
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 147
extern size_t strxfrm(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 149
 __attribute((__nonnull__(2))); 
# 28 "/usr/include/bits/types/__locale_t.h" 3
struct __locale_struct { 
# 31
struct __locale_data *__locales[13]; 
# 34
const unsigned short *__ctype_b; 
# 35
const int *__ctype_tolower; 
# 36
const int *__ctype_toupper; 
# 39
const char *__names[13]; 
# 40
}; 
# 42
typedef __locale_struct *__locale_t; 
# 24 "/usr/include/bits/types/locale_t.h" 3
typedef __locale_t locale_t; 
# 156 "/usr/include/string.h" 3
extern int strcoll_l(const char * __s1, const char * __s2, locale_t __l) throw()
# 157
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 3))); 
# 160
extern size_t strxfrm_l(char * __dest, const char * __src, size_t __n, locale_t __l) throw()
# 161
 __attribute((__nonnull__(2, 4))); 
# 167
extern char *strdup(const char * __s) throw()
# 168
 __attribute((__malloc__)) __attribute((__nonnull__(1))); 
# 175
extern char *strndup(const char * __string, size_t __n) throw()
# 176
 __attribute((__malloc__)) __attribute((__nonnull__(1))); 
# 204 "/usr/include/string.h" 3
extern "C++" {
# 206
extern char *strchr(char * __s, int __c) throw() __asm__("strchr")
# 207
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 208
extern const char *strchr(const char * __s, int __c) throw() __asm__("strchr")
# 209
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 224 "/usr/include/string.h" 3
}
# 231
extern "C++" {
# 233
extern char *strrchr(char * __s, int __c) throw() __asm__("strrchr")
# 234
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 235
extern const char *strrchr(const char * __s, int __c) throw() __asm__("strrchr")
# 236
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 251 "/usr/include/string.h" 3
}
# 261 "/usr/include/string.h" 3
extern "C++" char *strchrnul(char * __s, int __c) throw() __asm__("strchrnul")
# 262
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 263
extern "C++" const char *strchrnul(const char * __s, int __c) throw() __asm__("strchrnul")
# 264
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 273 "/usr/include/string.h" 3
extern size_t strcspn(const char * __s, const char * __reject) throw()
# 274
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 277
extern size_t strspn(const char * __s, const char * __accept) throw()
# 278
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 281
extern "C++" {
# 283
extern char *strpbrk(char * __s, const char * __accept) throw() __asm__("strpbrk")
# 284
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 285
extern const char *strpbrk(const char * __s, const char * __accept) throw() __asm__("strpbrk")
# 286
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 301 "/usr/include/string.h" 3
}
# 308
extern "C++" {
# 310
extern char *strstr(char * __haystack, const char * __needle) throw() __asm__("strstr")
# 311
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 312
extern const char *strstr(const char * __haystack, const char * __needle) throw() __asm__("strstr")
# 313
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 328 "/usr/include/string.h" 3
}
# 336
extern char *strtok(char *__restrict__ __s, const char *__restrict__ __delim) throw()
# 337
 __attribute((__nonnull__(2))); 
# 341
extern char *__strtok_r(char *__restrict__ __s, const char *__restrict__ __delim, char **__restrict__ __save_ptr) throw()
# 344
 __attribute((__nonnull__(2, 3))); 
# 346
extern char *strtok_r(char *__restrict__ __s, const char *__restrict__ __delim, char **__restrict__ __save_ptr) throw()
# 348
 __attribute((__nonnull__(2, 3))); 
# 354
extern "C++" char *strcasestr(char * __haystack, const char * __needle) throw() __asm__("strcasestr")
# 355
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 356
extern "C++" const char *strcasestr(const char * __haystack, const char * __needle) throw() __asm__("strcasestr")
# 358
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 369 "/usr/include/string.h" 3
extern void *memmem(const void * __haystack, size_t __haystacklen, const void * __needle, size_t __needlelen) throw()
# 371
 __attribute((__pure__)) __attribute((__nonnull__(1, 3))); 
# 375
extern void *__mempcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 377
 __attribute((__nonnull__(1, 2))); 
# 378
extern void *mempcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 380
 __attribute((__nonnull__(1, 2))); 
# 385
extern size_t strlen(const char * __s) throw()
# 386
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 391
extern size_t strnlen(const char * __string, size_t __maxlen) throw()
# 392
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 397
extern char *strerror(int __errnum) throw(); 
# 421 "/usr/include/string.h" 3
extern char *strerror_r(int __errnum, char * __buf, size_t __buflen) throw()
# 422
 __attribute((__nonnull__(2))); 
# 428
extern char *strerror_l(int __errnum, locale_t __l) throw(); 
# 30 "/usr/include/strings.h" 3
extern "C" {
# 34
extern int bcmp(const void * __s1, const void * __s2, size_t __n) throw()
# 35
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 38
extern void bcopy(const void * __src, void * __dest, size_t __n) throw()
# 39
 __attribute((__nonnull__(1, 2))); 
# 42
extern void bzero(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 46
extern "C++" {
# 48
extern char *index(char * __s, int __c) throw() __asm__("index")
# 49
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 50
extern const char *index(const char * __s, int __c) throw() __asm__("index")
# 51
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 66 "/usr/include/strings.h" 3
}
# 74
extern "C++" {
# 76
extern char *rindex(char * __s, int __c) throw() __asm__("rindex")
# 77
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 78
extern const char *rindex(const char * __s, int __c) throw() __asm__("rindex")
# 79
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 94 "/usr/include/strings.h" 3
}
# 104 "/usr/include/strings.h" 3
extern int ffs(int __i) throw() __attribute((const)); 
# 110
extern int ffsl(long __l) throw() __attribute((const)); 
# 111
__extension__ extern int ffsll(long long __ll) throw()
# 112
 __attribute((const)); 
# 116
extern int strcasecmp(const char * __s1, const char * __s2) throw()
# 117
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 120
extern int strncasecmp(const char * __s1, const char * __s2, size_t __n) throw()
# 121
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 128
extern int strcasecmp_l(const char * __s1, const char * __s2, locale_t __loc) throw()
# 129
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 3))); 
# 133
extern int strncasecmp_l(const char * __s1, const char * __s2, size_t __n, locale_t __loc) throw()
# 135
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 4))); 
# 138
}
# 436 "/usr/include/string.h" 3
extern void explicit_bzero(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 440
extern char *strsep(char **__restrict__ __stringp, const char *__restrict__ __delim) throw()
# 442
 __attribute((__nonnull__(1, 2))); 
# 447
extern char *strsignal(int __sig) throw(); 
# 450
extern char *__stpcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 451
 __attribute((__nonnull__(1, 2))); 
# 452
extern char *stpcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 453
 __attribute((__nonnull__(1, 2))); 
# 457
extern char *__stpncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 459
 __attribute((__nonnull__(1, 2))); 
# 460
extern char *stpncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 462
 __attribute((__nonnull__(1, 2))); 
# 467
extern int strverscmp(const char * __s1, const char * __s2) throw()
# 468
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 471
extern char *strfry(char * __string) throw() __attribute((__nonnull__(1))); 
# 474
extern void *memfrob(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 482
extern "C++" char *basename(char * __filename) throw() __asm__("basename")
# 483
 __attribute((__nonnull__(1))); 
# 484
extern "C++" const char *basename(const char * __filename) throw() __asm__("basename")
# 485
 __attribute((__nonnull__(1))); 
# 499 "/usr/include/string.h" 3
}
# 30 "/usr/include/bits/types.h" 3
typedef unsigned char __u_char; 
# 31
typedef unsigned short __u_short; 
# 32
typedef unsigned __u_int; 
# 33
typedef unsigned long __u_long; 
# 36
typedef signed char __int8_t; 
# 37
typedef unsigned char __uint8_t; 
# 38
typedef signed short __int16_t; 
# 39
typedef unsigned short __uint16_t; 
# 40
typedef signed int __int32_t; 
# 41
typedef unsigned __uint32_t; 
# 43
typedef signed long __int64_t; 
# 44
typedef unsigned long __uint64_t; 
# 51
typedef __int8_t __int_least8_t; 
# 52
typedef __uint8_t __uint_least8_t; 
# 53
typedef __int16_t __int_least16_t; 
# 54
typedef __uint16_t __uint_least16_t; 
# 55
typedef __int32_t __int_least32_t; 
# 56
typedef __uint32_t __uint_least32_t; 
# 57
typedef __int64_t __int_least64_t; 
# 58
typedef __uint64_t __uint_least64_t; 
# 62
typedef long __quad_t; 
# 63
typedef unsigned long __u_quad_t; 
# 71
typedef long __intmax_t; 
# 72
typedef unsigned long __uintmax_t; 
# 143 "/usr/include/bits/types.h" 3
typedef unsigned long __dev_t; 
# 144
typedef unsigned __uid_t; 
# 145
typedef unsigned __gid_t; 
# 146
typedef unsigned long __ino_t; 
# 147
typedef unsigned long __ino64_t; 
# 148
typedef unsigned __mode_t; 
# 149
typedef unsigned long __nlink_t; 
# 150
typedef long __off_t; 
# 151
typedef long __off64_t; 
# 152
typedef int __pid_t; 
# 153
typedef struct { int __val[2]; } __fsid_t; 
# 154
typedef long __clock_t; 
# 155
typedef unsigned long __rlim_t; 
# 156
typedef unsigned long __rlim64_t; 
# 157
typedef unsigned __id_t; 
# 158
typedef long __time_t; 
# 159
typedef unsigned __useconds_t; 
# 160
typedef long __suseconds_t; 
# 162
typedef int __daddr_t; 
# 163
typedef int __key_t; 
# 166
typedef int __clockid_t; 
# 169
typedef void *__timer_t; 
# 172
typedef long __blksize_t; 
# 177
typedef long __blkcnt_t; 
# 178
typedef long __blkcnt64_t; 
# 181
typedef unsigned long __fsblkcnt_t; 
# 182
typedef unsigned long __fsblkcnt64_t; 
# 185
typedef unsigned long __fsfilcnt_t; 
# 186
typedef unsigned long __fsfilcnt64_t; 
# 189
typedef long __fsword_t; 
# 191
typedef long __ssize_t; 
# 194
typedef long __syscall_slong_t; 
# 196
typedef unsigned long __syscall_ulong_t; 
# 200
typedef __off64_t __loff_t; 
# 201
typedef char *__caddr_t; 
# 204
typedef long __intptr_t; 
# 207
typedef unsigned __socklen_t; 
# 212
typedef int __sig_atomic_t; 
# 8 "/usr/include/bits/types/struct_timeval.h" 3
struct timeval { 
# 10
__time_t tv_sec; 
# 11
__suseconds_t tv_usec; 
# 12
}; 
# 26 "/usr/include/bits/timex.h" 3
struct timex { 
# 28
unsigned modes; 
# 29
__syscall_slong_t offset; 
# 30
__syscall_slong_t freq; 
# 31
__syscall_slong_t maxerror; 
# 32
__syscall_slong_t esterror; 
# 33
int status; 
# 34
__syscall_slong_t constant; 
# 35
__syscall_slong_t precision; 
# 36
__syscall_slong_t tolerance; 
# 37
timeval time; 
# 38
__syscall_slong_t tick; 
# 39
__syscall_slong_t ppsfreq; 
# 40
__syscall_slong_t jitter; 
# 41
int shift; 
# 42
__syscall_slong_t stabil; 
# 43
__syscall_slong_t jitcnt; 
# 44
__syscall_slong_t calcnt; 
# 45
__syscall_slong_t errcnt; 
# 46
__syscall_slong_t stbcnt; 
# 48
int tai; 
# 51
int:32; int:32; int:32; int:32; 
# 52
int:32; int:32; int:32; int:32; 
# 53
int:32; int:32; int:32; 
# 54
}; 
# 75 "/usr/include/bits/time.h" 3
extern "C" {
# 78
extern int clock_adjtime(__clockid_t __clock_id, timex * __utx) throw(); 
# 80
}
# 7 "/usr/include/bits/types/clock_t.h" 3
typedef __clock_t clock_t; 
# 7 "/usr/include/bits/types/time_t.h" 3
typedef __time_t time_t; 
# 7 "/usr/include/bits/types/struct_tm.h" 3
struct tm { 
# 9
int tm_sec; 
# 10
int tm_min; 
# 11
int tm_hour; 
# 12
int tm_mday; 
# 13
int tm_mon; 
# 14
int tm_year; 
# 15
int tm_wday; 
# 16
int tm_yday; 
# 17
int tm_isdst; 
# 20
long tm_gmtoff; 
# 21
const char *tm_zone; 
# 26
}; 
# 9 "/usr/include/bits/types/struct_timespec.h" 3
struct timespec { 
# 11
__time_t tv_sec; 
# 12
__syscall_slong_t tv_nsec; 
# 13
}; 
# 7 "/usr/include/bits/types/clockid_t.h" 3
typedef __clockid_t clockid_t; 
# 7 "/usr/include/bits/types/timer_t.h" 3
typedef __timer_t timer_t; 
# 8 "/usr/include/bits/types/struct_itimerspec.h" 3
struct itimerspec { 
# 10
timespec it_interval; 
# 11
timespec it_value; 
# 12
}; 
# 49 "/usr/include/time.h" 3
struct sigevent; 
# 54
typedef __pid_t pid_t; 
# 68 "/usr/include/time.h" 3
extern "C" {
# 72
extern clock_t clock() throw(); 
# 75
extern time_t time(time_t * __timer) throw(); 
# 78
extern double difftime(time_t __time1, time_t __time0) throw()
# 79
 __attribute((const)); 
# 82
extern time_t mktime(tm * __tp) throw(); 
# 88
extern size_t strftime(char *__restrict__ __s, size_t __maxsize, const char *__restrict__ __format, const tm *__restrict__ __tp) throw(); 
# 95
extern char *strptime(const char *__restrict__ __s, const char *__restrict__ __fmt, tm * __tp) throw(); 
# 104
extern size_t strftime_l(char *__restrict__ __s, size_t __maxsize, const char *__restrict__ __format, const tm *__restrict__ __tp, locale_t __loc) throw(); 
# 111
extern char *strptime_l(const char *__restrict__ __s, const char *__restrict__ __fmt, tm * __tp, locale_t __loc) throw(); 
# 119
extern tm *gmtime(const time_t * __timer) throw(); 
# 123
extern tm *localtime(const time_t * __timer) throw(); 
# 128
extern tm *gmtime_r(const time_t *__restrict__ __timer, tm *__restrict__ __tp) throw(); 
# 133
extern tm *localtime_r(const time_t *__restrict__ __timer, tm *__restrict__ __tp) throw(); 
# 139
extern char *asctime(const tm * __tp) throw(); 
# 142
extern char *ctime(const time_t * __timer) throw(); 
# 149
extern char *asctime_r(const tm *__restrict__ __tp, char *__restrict__ __buf) throw(); 
# 153
extern char *ctime_r(const time_t *__restrict__ __timer, char *__restrict__ __buf) throw(); 
# 159
extern char *__tzname[2]; 
# 160
extern int __daylight; 
# 161
extern long __timezone; 
# 166
extern char *tzname[2]; 
# 170
extern void tzset() throw(); 
# 174
extern int daylight; 
# 175
extern long timezone; 
# 181
extern int stime(const time_t * __when) throw(); 
# 196 "/usr/include/time.h" 3
extern time_t timegm(tm * __tp) throw(); 
# 199
extern time_t timelocal(tm * __tp) throw(); 
# 202
extern int dysize(int __year) throw() __attribute((const)); 
# 211 "/usr/include/time.h" 3
extern int nanosleep(const timespec * __requested_time, timespec * __remaining); 
# 216
extern int clock_getres(clockid_t __clock_id, timespec * __res) throw(); 
# 219
extern int clock_gettime(clockid_t __clock_id, timespec * __tp) throw(); 
# 222
extern int clock_settime(clockid_t __clock_id, const timespec * __tp) throw(); 
# 230
extern int clock_nanosleep(clockid_t __clock_id, int __flags, const timespec * __req, timespec * __rem); 
# 235
extern int clock_getcpuclockid(pid_t __pid, clockid_t * __clock_id) throw(); 
# 240
extern int timer_create(clockid_t __clock_id, sigevent *__restrict__ __evp, timer_t *__restrict__ __timerid) throw(); 
# 245
extern int timer_delete(timer_t __timerid) throw(); 
# 248
extern int timer_settime(timer_t __timerid, int __flags, const itimerspec *__restrict__ __value, itimerspec *__restrict__ __ovalue) throw(); 
# 253
extern int timer_gettime(timer_t __timerid, itimerspec * __value) throw(); 
# 257
extern int timer_getoverrun(timer_t __timerid) throw(); 
# 263
extern int timespec_get(timespec * __ts, int __base) throw()
# 264
 __attribute((__nonnull__(1))); 
# 280 "/usr/include/time.h" 3
extern int getdate_err; 
# 289 "/usr/include/time.h" 3
extern tm *getdate(const char * __string); 
# 303 "/usr/include/time.h" 3
extern int getdate_r(const char *__restrict__ __string, tm *__restrict__ __resbufp); 
# 307
}
# 88 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/common_functions.h" 3
extern "C" {
# 91
extern clock_t clock() throw(); 
# 96
extern void *memset(void *, int, size_t) throw(); 
# 97
extern void *memcpy(void *, const void *, size_t) throw(); 
# 99
}
# 121 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern "C" {
# 219 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern int abs(int a) throw(); 
# 227
extern long labs(long a) throw(); 
# 235
extern long long llabs(long long a) throw(); 
# 285 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double fabs(double x) throw(); 
# 328 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float fabsf(float x) throw(); 
# 338 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern inline int min(const int a, const int b); 
# 345
extern inline unsigned umin(const unsigned a, const unsigned b); 
# 352
extern inline long long llmin(const long long a, const long long b); 
# 359
extern inline unsigned long long ullmin(const unsigned long long a, const unsigned long long b); 
# 380 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float fminf(float x, float y) throw(); 
# 400 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double fmin(double x, double y) throw(); 
# 413 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern inline int max(const int a, const int b); 
# 421
extern inline unsigned umax(const unsigned a, const unsigned b); 
# 428
extern inline long long llmax(const long long a, const long long b); 
# 435
extern inline unsigned long long ullmax(const unsigned long long a, const unsigned long long b); 
# 456 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float fmaxf(float x, float y) throw(); 
# 476 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double fmax(double, double) throw(); 
# 520 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double sin(double x) throw(); 
# 553 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double cos(double x) throw(); 
# 572 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern void sincos(double x, double * sptr, double * cptr) throw(); 
# 588 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern void sincosf(float x, float * sptr, float * cptr) throw(); 
# 633 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double tan(double x) throw(); 
# 702 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double sqrt(double x) throw(); 
# 774 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double rsqrt(double x); 
# 844 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float rsqrtf(float x); 
# 900 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double log2(double x) throw(); 
# 965 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double exp2(double x) throw(); 
# 1030 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float exp2f(float x) throw(); 
# 1097 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double exp10(double x) throw(); 
# 1160 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float exp10f(float x) throw(); 
# 1253 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double expm1(double x) throw(); 
# 1345 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float expm1f(float x) throw(); 
# 1401 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float log2f(float x) throw(); 
# 1455 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double log10(double x) throw(); 
# 1525 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double log(double x) throw(); 
# 1621 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double log1p(double x) throw(); 
# 1720 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float log1pf(float x) throw(); 
# 1784 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double floor(double x) throw(); 
# 1863 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double exp(double x) throw(); 
# 1904 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double cosh(double x) throw(); 
# 1954 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double sinh(double x) throw(); 
# 2004 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double tanh(double x) throw(); 
# 2059 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double acosh(double x) throw(); 
# 2117 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float acoshf(float x) throw(); 
# 2170 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double asinh(double x) throw(); 
# 2223 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float asinhf(float x) throw(); 
# 2277 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double atanh(double x) throw(); 
# 2331 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float atanhf(float x) throw(); 
# 2380 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double ldexp(double x, int exp) throw(); 
# 2426 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float ldexpf(float x, int exp) throw(); 
# 2478 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double logb(double x) throw(); 
# 2533 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float logbf(float x) throw(); 
# 2573 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern int ilogb(double x) throw(); 
# 2613 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern int ilogbf(float x) throw(); 
# 2689 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double scalbn(double x, int n) throw(); 
# 2765 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float scalbnf(float x, int n) throw(); 
# 2841 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double scalbln(double x, long n) throw(); 
# 2917 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float scalblnf(float x, long n) throw(); 
# 2994 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double frexp(double x, int * nptr) throw(); 
# 3068 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float frexpf(float x, int * nptr) throw(); 
# 3120 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double round(double x) throw(); 
# 3175 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float roundf(float x) throw(); 
# 3193 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern long lround(double x) throw(); 
# 3211 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern long lroundf(float x) throw(); 
# 3229 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern long long llround(double x) throw(); 
# 3247 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern long long llroundf(float x) throw(); 
# 3375 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float rintf(float x) throw(); 
# 3392 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern long lrint(double x) throw(); 
# 3409 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern long lrintf(float x) throw(); 
# 3426 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern long long llrint(double x) throw(); 
# 3443 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern long long llrintf(float x) throw(); 
# 3496 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double nearbyint(double x) throw(); 
# 3549 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float nearbyintf(float x) throw(); 
# 3611 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double ceil(double x) throw(); 
# 3661 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double trunc(double x) throw(); 
# 3714 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float truncf(float x) throw(); 
# 3740 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double fdim(double x, double y) throw(); 
# 3766 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float fdimf(float x, float y) throw(); 
# 4066 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double atan2(double y, double x) throw(); 
# 4137 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double atan(double x) throw(); 
# 4160 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double acos(double x) throw(); 
# 4211 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double asin(double x) throw(); 
# 4279 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double hypot(double x, double y) throw(); 
# 4402 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float hypotf(float x, float y) throw(); 
# 5188 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double cbrt(double x) throw(); 
# 5274 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float cbrtf(float x) throw(); 
# 5329 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double rcbrt(double x); 
# 5379 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float rcbrtf(float x); 
# 5439 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double sinpi(double x); 
# 5499 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float sinpif(float x); 
# 5551 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double cospi(double x); 
# 5603 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float cospif(float x); 
# 5633 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern void sincospi(double x, double * sptr, double * cptr); 
# 5663 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern void sincospif(float x, float * sptr, float * cptr); 
# 5996 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double pow(double x, double y) throw(); 
# 6052 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double modf(double x, double * iptr) throw(); 
# 6111 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double fmod(double x, double y) throw(); 
# 6207 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double remainder(double x, double y) throw(); 
# 6306 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float remainderf(float x, float y) throw(); 
# 6378 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double remquo(double x, double y, int * quo) throw(); 
# 6450 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float remquof(float x, float y, int * quo) throw(); 
# 6491 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double j0(double x) throw(); 
# 6533 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float j0f(float x) throw(); 
# 6602 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double j1(double x) throw(); 
# 6671 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float j1f(float x) throw(); 
# 6714 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double jn(int n, double x) throw(); 
# 6757 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float jnf(int n, float x) throw(); 
# 6818 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double y0(double x) throw(); 
# 6879 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float y0f(float x) throw(); 
# 6940 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double y1(double x) throw(); 
# 7001 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float y1f(float x) throw(); 
# 7064 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double yn(int n, double x) throw(); 
# 7127 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float ynf(int n, float x) throw(); 
# 7316 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double erf(double x) throw(); 
# 7398 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float erff(float x) throw(); 
# 7470 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double erfinv(double x); 
# 7535 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float erfinvf(float x); 
# 7574 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double erfc(double x) throw(); 
# 7612 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float erfcf(float x) throw(); 
# 7729 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double lgamma(double x) throw(); 
# 7791 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double erfcinv(double x); 
# 7846 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float erfcinvf(float x); 
# 7914 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double normcdfinv(double x); 
# 7982 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float normcdfinvf(float x); 
# 8025 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double normcdf(double x); 
# 8068 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float normcdff(float x); 
# 8132 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double erfcx(double x); 
# 8196 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float erfcxf(float x); 
# 8315 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float lgammaf(float x) throw(); 
# 8413 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double tgamma(double x) throw(); 
# 8511 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float tgammaf(float x) throw(); 
# 8524 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double copysign(double x, double y) throw(); 
# 8537 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float copysignf(float x, float y) throw(); 
# 8556 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double nextafter(double x, double y) throw(); 
# 8575 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float nextafterf(float x, float y) throw(); 
# 8591 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double nan(const char * tagp) throw(); 
# 8607 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float nanf(const char * tagp) throw(); 
# 8614
extern int __isinff(float) throw(); 
# 8615
extern int __isnanf(float) throw(); 
# 8625 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern int __finite(double) throw(); 
# 8626
extern int __finitef(float) throw(); 
# 8627
extern int __signbit(double) throw(); 
# 8628
extern int __isnan(double) throw(); 
# 8629
extern int __isinf(double) throw(); 
# 8632
extern int __signbitf(float) throw(); 
# 8791 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern double fma(double x, double y, double z) throw(); 
# 8949 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float fmaf(float x, float y, float z) throw(); 
# 8960 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern int __signbitl(long double) throw(); 
# 8966
extern int __finitel(long double) throw(); 
# 8967
extern int __isinfl(long double) throw(); 
# 8968
extern int __isnanl(long double) throw(); 
# 9018 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float acosf(float x) throw(); 
# 9077 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float asinf(float x) throw(); 
# 9157 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float atanf(float x) throw(); 
# 9454 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float atan2f(float y, float x) throw(); 
# 9488 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float cosf(float x) throw(); 
# 9530 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float sinf(float x) throw(); 
# 9572 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float tanf(float x) throw(); 
# 9613 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float coshf(float x) throw(); 
# 9663 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float sinhf(float x) throw(); 
# 9713 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float tanhf(float x) throw(); 
# 9765 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float logf(float x) throw(); 
# 9845 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float expf(float x) throw(); 
# 9897 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float log10f(float x) throw(); 
# 9952 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float modff(float x, float * iptr) throw(); 
# 10282 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float powf(float x, float y) throw(); 
# 10351 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float sqrtf(float x) throw(); 
# 10410 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float ceilf(float x) throw(); 
# 10471 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float floorf(float x) throw(); 
# 10529 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern float fmodf(float x, float y) throw(); 
# 10544 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
}
# 2198 "/usr/include/c++/8/x86_64-redhat-linux/bits/c++config.h" 3
namespace std { 
# 2200
typedef unsigned long size_t; 
# 2201
typedef long ptrdiff_t; 
# 2204
typedef __decltype((nullptr)) nullptr_t; 
# 2206
}
# 2220 "/usr/include/c++/8/x86_64-redhat-linux/bits/c++config.h" 3
namespace std { 
# 2222
inline namespace __cxx11 __attribute((__abi_tag__("cxx11"))) { }
# 2223
}
# 2224
namespace __gnu_cxx { 
# 2226
inline namespace __cxx11 __attribute((__abi_tag__("cxx11"))) { }
# 2227
}
# 67 "/usr/include/c++/8/bits/cpp_type_traits.h" 3
extern "C++" {
# 69
namespace std __attribute((__visibility__("default"))) { 
# 73
struct __true_type { }; 
# 74
struct __false_type { }; 
# 76
template< bool > 
# 77
struct __truth_type { 
# 78
typedef __false_type __type; }; 
# 81
template<> struct __truth_type< true>  { 
# 82
typedef __true_type __type; }; 
# 86
template< class _Sp, class _Tp> 
# 87
struct __traitor { 
# 89
enum { __value = ((bool)_Sp::__value) || ((bool)_Tp::__value)}; 
# 90
typedef typename __truth_type< __value> ::__type __type; 
# 91
}; 
# 94
template< class , class > 
# 95
struct __are_same { 
# 97
enum { __value}; 
# 98
typedef __false_type __type; 
# 99
}; 
# 101
template< class _Tp> 
# 102
struct __are_same< _Tp, _Tp>  { 
# 104
enum { __value = 1}; 
# 105
typedef __true_type __type; 
# 106
}; 
# 109
template< class _Tp> 
# 110
struct __is_void { 
# 112
enum { __value}; 
# 113
typedef __false_type __type; 
# 114
}; 
# 117
template<> struct __is_void< void>  { 
# 119
enum { __value = 1}; 
# 120
typedef __true_type __type; 
# 121
}; 
# 126
template< class _Tp> 
# 127
struct __is_integer { 
# 129
enum { __value}; 
# 130
typedef __false_type __type; 
# 131
}; 
# 138
template<> struct __is_integer< bool>  { 
# 140
enum { __value = 1}; 
# 141
typedef __true_type __type; 
# 142
}; 
# 145
template<> struct __is_integer< char>  { 
# 147
enum { __value = 1}; 
# 148
typedef __true_type __type; 
# 149
}; 
# 152
template<> struct __is_integer< signed char>  { 
# 154
enum { __value = 1}; 
# 155
typedef __true_type __type; 
# 156
}; 
# 159
template<> struct __is_integer< unsigned char>  { 
# 161
enum { __value = 1}; 
# 162
typedef __true_type __type; 
# 163
}; 
# 167
template<> struct __is_integer< wchar_t>  { 
# 169
enum { __value = 1}; 
# 170
typedef __true_type __type; 
# 171
}; 
# 176
template<> struct __is_integer< char16_t>  { 
# 178
enum { __value = 1}; 
# 179
typedef __true_type __type; 
# 180
}; 
# 183
template<> struct __is_integer< char32_t>  { 
# 185
enum { __value = 1}; 
# 186
typedef __true_type __type; 
# 187
}; 
# 191
template<> struct __is_integer< short>  { 
# 193
enum { __value = 1}; 
# 194
typedef __true_type __type; 
# 195
}; 
# 198
template<> struct __is_integer< unsigned short>  { 
# 200
enum { __value = 1}; 
# 201
typedef __true_type __type; 
# 202
}; 
# 205
template<> struct __is_integer< int>  { 
# 207
enum { __value = 1}; 
# 208
typedef __true_type __type; 
# 209
}; 
# 212
template<> struct __is_integer< unsigned>  { 
# 214
enum { __value = 1}; 
# 215
typedef __true_type __type; 
# 216
}; 
# 219
template<> struct __is_integer< long>  { 
# 221
enum { __value = 1}; 
# 222
typedef __true_type __type; 
# 223
}; 
# 226
template<> struct __is_integer< unsigned long>  { 
# 228
enum { __value = 1}; 
# 229
typedef __true_type __type; 
# 230
}; 
# 233
template<> struct __is_integer< long long>  { 
# 235
enum { __value = 1}; 
# 236
typedef __true_type __type; 
# 237
}; 
# 240
template<> struct __is_integer< unsigned long long>  { 
# 242
enum { __value = 1}; 
# 243
typedef __true_type __type; 
# 244
}; 
# 261 "/usr/include/c++/8/bits/cpp_type_traits.h" 3
template<> struct __is_integer< __int128>  { enum { __value = 1}; typedef __true_type __type; }; template<> struct __is_integer< unsigned __int128>  { enum { __value = 1}; typedef __true_type __type; }; 
# 278 "/usr/include/c++/8/bits/cpp_type_traits.h" 3
template< class _Tp> 
# 279
struct __is_floating { 
# 281
enum { __value}; 
# 282
typedef __false_type __type; 
# 283
}; 
# 287
template<> struct __is_floating< float>  { 
# 289
enum { __value = 1}; 
# 290
typedef __true_type __type; 
# 291
}; 
# 294
template<> struct __is_floating< double>  { 
# 296
enum { __value = 1}; 
# 297
typedef __true_type __type; 
# 298
}; 
# 301
template<> struct __is_floating< long double>  { 
# 303
enum { __value = 1}; 
# 304
typedef __true_type __type; 
# 305
}; 
# 310
template< class _Tp> 
# 311
struct __is_pointer { 
# 313
enum { __value}; 
# 314
typedef __false_type __type; 
# 315
}; 
# 317
template< class _Tp> 
# 318
struct __is_pointer< _Tp *>  { 
# 320
enum { __value = 1}; 
# 321
typedef __true_type __type; 
# 322
}; 
# 327
template< class _Tp> 
# 328
struct __is_arithmetic : public __traitor< __is_integer< _Tp> , __is_floating< _Tp> >  { 
# 330
}; 
# 335
template< class _Tp> 
# 336
struct __is_scalar : public __traitor< __is_arithmetic< _Tp> , __is_pointer< _Tp> >  { 
# 338
}; 
# 343
template< class _Tp> 
# 344
struct __is_char { 
# 346
enum { __value}; 
# 347
typedef __false_type __type; 
# 348
}; 
# 351
template<> struct __is_char< char>  { 
# 353
enum { __value = 1}; 
# 354
typedef __true_type __type; 
# 355
}; 
# 359
template<> struct __is_char< wchar_t>  { 
# 361
enum { __value = 1}; 
# 362
typedef __true_type __type; 
# 363
}; 
# 366
template< class _Tp> 
# 367
struct __is_byte { 
# 369
enum { __value}; 
# 370
typedef __false_type __type; 
# 371
}; 
# 374
template<> struct __is_byte< char>  { 
# 376
enum { __value = 1}; 
# 377
typedef __true_type __type; 
# 378
}; 
# 381
template<> struct __is_byte< signed char>  { 
# 383
enum { __value = 1}; 
# 384
typedef __true_type __type; 
# 385
}; 
# 388
template<> struct __is_byte< unsigned char>  { 
# 390
enum { __value = 1}; 
# 391
typedef __true_type __type; 
# 392
}; 
# 408 "/usr/include/c++/8/bits/cpp_type_traits.h" 3
template< class _Tp> 
# 409
struct __is_move_iterator { 
# 411
enum { __value}; 
# 412
typedef __false_type __type; 
# 413
}; 
# 417
template< class _Iterator> inline _Iterator 
# 419
__miter_base(_Iterator __it) 
# 420
{ return __it; } 
# 423
}
# 424
}
# 37 "/usr/include/c++/8/ext/type_traits.h" 3
extern "C++" {
# 39
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 44
template< bool , class > 
# 45
struct __enable_if { 
# 46
}; 
# 48
template< class _Tp> 
# 49
struct __enable_if< true, _Tp>  { 
# 50
typedef _Tp __type; }; 
# 54
template< bool _Cond, class _Iftrue, class _Iffalse> 
# 55
struct __conditional_type { 
# 56
typedef _Iftrue __type; }; 
# 58
template< class _Iftrue, class _Iffalse> 
# 59
struct __conditional_type< false, _Iftrue, _Iffalse>  { 
# 60
typedef _Iffalse __type; }; 
# 64
template< class _Tp> 
# 65
struct __add_unsigned { 
# 68
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp>  __if_type; 
# 71
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type; 
# 72
}; 
# 75
template<> struct __add_unsigned< char>  { 
# 76
typedef unsigned char __type; }; 
# 79
template<> struct __add_unsigned< signed char>  { 
# 80
typedef unsigned char __type; }; 
# 83
template<> struct __add_unsigned< short>  { 
# 84
typedef unsigned short __type; }; 
# 87
template<> struct __add_unsigned< int>  { 
# 88
typedef unsigned __type; }; 
# 91
template<> struct __add_unsigned< long>  { 
# 92
typedef unsigned long __type; }; 
# 95
template<> struct __add_unsigned< long long>  { 
# 96
typedef unsigned long long __type; }; 
# 100
template<> struct __add_unsigned< bool> ; 
# 103
template<> struct __add_unsigned< wchar_t> ; 
# 107
template< class _Tp> 
# 108
struct __remove_unsigned { 
# 111
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp>  __if_type; 
# 114
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type; 
# 115
}; 
# 118
template<> struct __remove_unsigned< char>  { 
# 119
typedef signed char __type; }; 
# 122
template<> struct __remove_unsigned< unsigned char>  { 
# 123
typedef signed char __type; }; 
# 126
template<> struct __remove_unsigned< unsigned short>  { 
# 127
typedef short __type; }; 
# 130
template<> struct __remove_unsigned< unsigned>  { 
# 131
typedef int __type; }; 
# 134
template<> struct __remove_unsigned< unsigned long>  { 
# 135
typedef long __type; }; 
# 138
template<> struct __remove_unsigned< unsigned long long>  { 
# 139
typedef long long __type; }; 
# 143
template<> struct __remove_unsigned< bool> ; 
# 146
template<> struct __remove_unsigned< wchar_t> ; 
# 150
template< class _Type> inline bool 
# 152
__is_null_pointer(_Type *__ptr) 
# 153
{ return __ptr == 0; } 
# 155
template< class _Type> inline bool 
# 157
__is_null_pointer(_Type) 
# 158
{ return false; } 
# 162
inline bool __is_null_pointer(std::nullptr_t) 
# 163
{ return true; } 
# 167
template< class _Tp, bool  = std::template __is_integer< _Tp> ::__value> 
# 168
struct __promote { 
# 169
typedef double __type; }; 
# 174
template< class _Tp> 
# 175
struct __promote< _Tp, false>  { 
# 176
}; 
# 179
template<> struct __promote< long double>  { 
# 180
typedef long double __type; }; 
# 183
template<> struct __promote< double>  { 
# 184
typedef double __type; }; 
# 187
template<> struct __promote< float>  { 
# 188
typedef float __type; }; 
# 190
template< class _Tp, class _Up, class 
# 191
_Tp2 = typename __promote< _Tp> ::__type, class 
# 192
_Up2 = typename __promote< _Up> ::__type> 
# 193
struct __promote_2 { 
# 195
typedef __typeof__(_Tp2() + _Up2()) __type; 
# 196
}; 
# 198
template< class _Tp, class _Up, class _Vp, class 
# 199
_Tp2 = typename __promote< _Tp> ::__type, class 
# 200
_Up2 = typename __promote< _Up> ::__type, class 
# 201
_Vp2 = typename __promote< _Vp> ::__type> 
# 202
struct __promote_3 { 
# 204
typedef __typeof__((_Tp2() + _Up2()) + _Vp2()) __type; 
# 205
}; 
# 207
template< class _Tp, class _Up, class _Vp, class _Wp, class 
# 208
_Tp2 = typename __promote< _Tp> ::__type, class 
# 209
_Up2 = typename __promote< _Up> ::__type, class 
# 210
_Vp2 = typename __promote< _Vp> ::__type, class 
# 211
_Wp2 = typename __promote< _Wp> ::__type> 
# 212
struct __promote_4 { 
# 214
typedef __typeof__(((_Tp2() + _Up2()) + _Vp2()) + _Wp2()) __type; 
# 215
}; 
# 218
}
# 219
}
# 34 "/usr/include/math.h" 3
extern "C" {
# 74 "/usr/include/bits/floatn.h" 3
typedef float __complex__ __cfloat128 __attribute((__mode__(__TC__))); 
# 86 "/usr/include/bits/floatn.h" 3
typedef __float128 _Float128; 
# 214 "/usr/include/bits/floatn-common.h" 3
typedef float _Float32; 
# 251 "/usr/include/bits/floatn-common.h" 3
typedef double _Float64; 
# 268 "/usr/include/bits/floatn-common.h" 3
typedef double _Float32x; 
# 285 "/usr/include/bits/floatn-common.h" 3
typedef long double _Float64x; 
# 149 "/usr/include/math.h" 3
typedef float float_t; 
# 150
typedef double double_t; 
# 238 "/usr/include/math.h" 3
enum { 
# 239
FP_INT_UPWARD, 
# 242
FP_INT_DOWNWARD, 
# 245
FP_INT_TOWARDZERO, 
# 248
FP_INT_TONEARESTFROMZERO, 
# 251
FP_INT_TONEAREST
# 254
}; 
# 21 "/usr/include/bits/mathcalls-helper-functions.h" 3
extern int __fpclassify(double __value) throw()
# 22
 __attribute((const)); 
# 25
extern int __signbit(double __value) throw()
# 26
 __attribute((const)); 
# 30
extern int __isinf(double __value) throw() __attribute((const)); 
# 33
extern int __finite(double __value) throw() __attribute((const)); 
# 36
extern int __isnan(double __value) throw() __attribute((const)); 
# 39
extern int __iseqsig(double __x, double __y) throw(); 
# 42
extern int __issignaling(double __value) throw()
# 43
 __attribute((const)); 
# 53 "/usr/include/bits/mathcalls.h" 3
extern double acos(double __x) throw(); extern double __acos(double __x) throw(); 
# 55
extern double asin(double __x) throw(); extern double __asin(double __x) throw(); 
# 57
extern double atan(double __x) throw(); extern double __atan(double __x) throw(); 
# 59
extern double atan2(double __y, double __x) throw(); extern double __atan2(double __y, double __x) throw(); 
# 62
extern double cos(double __x) throw(); extern double __cos(double __x) throw(); 
# 64
extern double sin(double __x) throw(); extern double __sin(double __x) throw(); 
# 66
extern double tan(double __x) throw(); extern double __tan(double __x) throw(); 
# 71
extern double cosh(double __x) throw(); extern double __cosh(double __x) throw(); 
# 73
extern double sinh(double __x) throw(); extern double __sinh(double __x) throw(); 
# 75
extern double tanh(double __x) throw(); extern double __tanh(double __x) throw(); 
# 79
extern void sincos(double __x, double * __sinx, double * __cosx) throw(); extern void __sincos(double __x, double * __sinx, double * __cosx) throw(); 
# 85
extern double acosh(double __x) throw(); extern double __acosh(double __x) throw(); 
# 87
extern double asinh(double __x) throw(); extern double __asinh(double __x) throw(); 
# 89
extern double atanh(double __x) throw(); extern double __atanh(double __x) throw(); 
# 95
extern double exp(double __x) throw(); extern double __exp(double __x) throw(); 
# 98
extern double frexp(double __x, int * __exponent) throw(); extern double __frexp(double __x, int * __exponent) throw(); 
# 101
extern double ldexp(double __x, int __exponent) throw(); extern double __ldexp(double __x, int __exponent) throw(); 
# 104
extern double log(double __x) throw(); extern double __log(double __x) throw(); 
# 107
extern double log10(double __x) throw(); extern double __log10(double __x) throw(); 
# 110
extern double modf(double __x, double * __iptr) throw(); extern double __modf(double __x, double * __iptr) throw() __attribute((__nonnull__(2))); 
# 114
extern double exp10(double __x) throw(); extern double __exp10(double __x) throw(); 
# 119
extern double expm1(double __x) throw(); extern double __expm1(double __x) throw(); 
# 122
extern double log1p(double __x) throw(); extern double __log1p(double __x) throw(); 
# 125
extern double logb(double __x) throw(); extern double __logb(double __x) throw(); 
# 130
extern double exp2(double __x) throw(); extern double __exp2(double __x) throw(); 
# 133
extern double log2(double __x) throw(); extern double __log2(double __x) throw(); 
# 140
extern double pow(double __x, double __y) throw(); extern double __pow(double __x, double __y) throw(); 
# 143
extern double sqrt(double __x) throw(); extern double __sqrt(double __x) throw(); 
# 147
extern double hypot(double __x, double __y) throw(); extern double __hypot(double __x, double __y) throw(); 
# 152
extern double cbrt(double __x) throw(); extern double __cbrt(double __x) throw(); 
# 159
extern double ceil(double __x) throw() __attribute((const)); extern double __ceil(double __x) throw() __attribute((const)); 
# 162
extern double fabs(double __x) throw() __attribute((const)); extern double __fabs(double __x) throw() __attribute((const)); 
# 165
extern double floor(double __x) throw() __attribute((const)); extern double __floor(double __x) throw() __attribute((const)); 
# 168
extern double fmod(double __x, double __y) throw(); extern double __fmod(double __x, double __y) throw(); 
# 182 "/usr/include/bits/mathcalls.h" 3
extern int finite(double __value) throw() __attribute((const)); 
# 185
extern double drem(double __x, double __y) throw(); extern double __drem(double __x, double __y) throw(); 
# 189
extern double significand(double __x) throw(); extern double __significand(double __x) throw(); 
# 196
extern double copysign(double __x, double __y) throw() __attribute((const)); extern double __copysign(double __x, double __y) throw() __attribute((const)); 
# 201
extern double nan(const char * __tagb) throw(); extern double __nan(const char * __tagb) throw(); 
# 217 "/usr/include/bits/mathcalls.h" 3
extern double j0(double) throw(); extern double __j0(double) throw(); 
# 218
extern double j1(double) throw(); extern double __j1(double) throw(); 
# 219
extern double jn(int, double) throw(); extern double __jn(int, double) throw(); 
# 220
extern double y0(double) throw(); extern double __y0(double) throw(); 
# 221
extern double y1(double) throw(); extern double __y1(double) throw(); 
# 222
extern double yn(int, double) throw(); extern double __yn(int, double) throw(); 
# 228
extern double erf(double) throw(); extern double __erf(double) throw(); 
# 229
extern double erfc(double) throw(); extern double __erfc(double) throw(); 
# 230
extern double lgamma(double) throw(); extern double __lgamma(double) throw(); 
# 235
extern double tgamma(double) throw(); extern double __tgamma(double) throw(); 
# 241
extern double gamma(double) throw(); extern double __gamma(double) throw(); 
# 249
extern double lgamma_r(double, int * __signgamp) throw(); extern double __lgamma_r(double, int * __signgamp) throw(); 
# 256
extern double rint(double __x) throw(); extern double __rint(double __x) throw(); 
# 259
extern double nextafter(double __x, double __y) throw(); extern double __nextafter(double __x, double __y) throw(); 
# 261
extern double nexttoward(double __x, long double __y) throw(); extern double __nexttoward(double __x, long double __y) throw(); 
# 266
extern double nextdown(double __x) throw(); extern double __nextdown(double __x) throw(); 
# 268
extern double nextup(double __x) throw(); extern double __nextup(double __x) throw(); 
# 272
extern double remainder(double __x, double __y) throw(); extern double __remainder(double __x, double __y) throw(); 
# 276
extern double scalbn(double __x, int __n) throw(); extern double __scalbn(double __x, int __n) throw(); 
# 280
extern int ilogb(double __x) throw(); extern int __ilogb(double __x) throw(); 
# 285
extern long llogb(double __x) throw(); extern long __llogb(double __x) throw(); 
# 290
extern double scalbln(double __x, long __n) throw(); extern double __scalbln(double __x, long __n) throw(); 
# 294
extern double nearbyint(double __x) throw(); extern double __nearbyint(double __x) throw(); 
# 298
extern double round(double __x) throw() __attribute((const)); extern double __round(double __x) throw() __attribute((const)); 
# 302
extern double trunc(double __x) throw() __attribute((const)); extern double __trunc(double __x) throw() __attribute((const)); 
# 307
extern double remquo(double __x, double __y, int * __quo) throw(); extern double __remquo(double __x, double __y, int * __quo) throw(); 
# 314
extern long lrint(double __x) throw(); extern long __lrint(double __x) throw(); 
# 316
__extension__ extern long long llrint(double __x) throw(); extern long long __llrint(double __x) throw(); 
# 320
extern long lround(double __x) throw(); extern long __lround(double __x) throw(); 
# 322
__extension__ extern long long llround(double __x) throw(); extern long long __llround(double __x) throw(); 
# 326
extern double fdim(double __x, double __y) throw(); extern double __fdim(double __x, double __y) throw(); 
# 329
extern double fmax(double __x, double __y) throw() __attribute((const)); extern double __fmax(double __x, double __y) throw() __attribute((const)); 
# 332
extern double fmin(double __x, double __y) throw() __attribute((const)); extern double __fmin(double __x, double __y) throw() __attribute((const)); 
# 335
extern double fma(double __x, double __y, double __z) throw(); extern double __fma(double __x, double __y, double __z) throw(); 
# 340
extern double roundeven(double __x) throw() __attribute((const)); extern double __roundeven(double __x) throw() __attribute((const)); 
# 344
extern __intmax_t fromfp(double __x, int __round, unsigned __width) throw(); extern __intmax_t __fromfp(double __x, int __round, unsigned __width) throw(); 
# 349
extern __uintmax_t ufromfp(double __x, int __round, unsigned __width) throw(); extern __uintmax_t __ufromfp(double __x, int __round, unsigned __width) throw(); 
# 355
extern __intmax_t fromfpx(double __x, int __round, unsigned __width) throw(); extern __intmax_t __fromfpx(double __x, int __round, unsigned __width) throw(); 
# 361
extern __uintmax_t ufromfpx(double __x, int __round, unsigned __width) throw(); extern __uintmax_t __ufromfpx(double __x, int __round, unsigned __width) throw(); 
# 365
extern double fmaxmag(double __x, double __y) throw() __attribute((const)); extern double __fmaxmag(double __x, double __y) throw() __attribute((const)); 
# 368
extern double fminmag(double __x, double __y) throw() __attribute((const)); extern double __fminmag(double __x, double __y) throw() __attribute((const)); 
# 371
extern int totalorder(double __x, double __y) throw()
# 372
 __attribute((const)); 
# 375
extern int totalordermag(double __x, double __y) throw()
# 376
 __attribute((const)); 
# 379
extern int canonicalize(double * __cx, const double * __x) throw(); 
# 382
extern double getpayload(const double * __x) throw(); extern double __getpayload(const double * __x) throw(); 
# 385
extern int setpayload(double * __x, double __payload) throw(); 
# 388
extern int setpayloadsig(double * __x, double __payload) throw(); 
# 396
extern double scalb(double __x, double __n) throw(); extern double __scalb(double __x, double __n) throw(); 
# 21 "/usr/include/bits/mathcalls-helper-functions.h" 3
extern int __fpclassifyf(float __value) throw()
# 22
 __attribute((const)); 
# 25
extern int __signbitf(float __value) throw()
# 26
 __attribute((const)); 
# 30
extern int __isinff(float __value) throw() __attribute((const)); 
# 33
extern int __finitef(float __value) throw() __attribute((const)); 
# 36
extern int __isnanf(float __value) throw() __attribute((const)); 
# 39
extern int __iseqsigf(float __x, float __y) throw(); 
# 42
extern int __issignalingf(float __value) throw()
# 43
 __attribute((const)); 
# 53 "/usr/include/bits/mathcalls.h" 3
extern float acosf(float __x) throw(); extern float __acosf(float __x) throw(); 
# 55
extern float asinf(float __x) throw(); extern float __asinf(float __x) throw(); 
# 57
extern float atanf(float __x) throw(); extern float __atanf(float __x) throw(); 
# 59
extern float atan2f(float __y, float __x) throw(); extern float __atan2f(float __y, float __x) throw(); 
# 62
extern float cosf(float __x) throw(); 
# 64
extern float sinf(float __x) throw(); 
# 66
extern float tanf(float __x) throw(); 
# 71
extern float coshf(float __x) throw(); extern float __coshf(float __x) throw(); 
# 73
extern float sinhf(float __x) throw(); extern float __sinhf(float __x) throw(); 
# 75
extern float tanhf(float __x) throw(); extern float __tanhf(float __x) throw(); 
# 79
extern void sincosf(float __x, float * __sinx, float * __cosx) throw(); 
# 85
extern float acoshf(float __x) throw(); extern float __acoshf(float __x) throw(); 
# 87
extern float asinhf(float __x) throw(); extern float __asinhf(float __x) throw(); 
# 89
extern float atanhf(float __x) throw(); extern float __atanhf(float __x) throw(); 
# 95
extern float expf(float __x) throw(); 
# 98
extern float frexpf(float __x, int * __exponent) throw(); extern float __frexpf(float __x, int * __exponent) throw(); 
# 101
extern float ldexpf(float __x, int __exponent) throw(); extern float __ldexpf(float __x, int __exponent) throw(); 
# 104
extern float logf(float __x) throw(); 
# 107
extern float log10f(float __x) throw(); 
# 110
extern float modff(float __x, float * __iptr) throw(); extern float __modff(float __x, float * __iptr) throw() __attribute((__nonnull__(2))); 
# 114
extern float exp10f(float __x) throw(); 
# 119
extern float expm1f(float __x) throw(); extern float __expm1f(float __x) throw(); 
# 122
extern float log1pf(float __x) throw(); extern float __log1pf(float __x) throw(); 
# 125
extern float logbf(float __x) throw(); extern float __logbf(float __x) throw(); 
# 130
extern float exp2f(float __x) throw(); extern float __exp2f(float __x) throw(); 
# 133
extern float log2f(float __x) throw(); 
# 140
extern float powf(float __x, float __y) throw(); 
# 143
extern float sqrtf(float __x) throw(); extern float __sqrtf(float __x) throw(); 
# 147
extern float hypotf(float __x, float __y) throw(); extern float __hypotf(float __x, float __y) throw(); 
# 152
extern float cbrtf(float __x) throw(); extern float __cbrtf(float __x) throw(); 
# 159
extern float ceilf(float __x) throw() __attribute((const)); extern float __ceilf(float __x) throw() __attribute((const)); 
# 162
extern float fabsf(float __x) throw() __attribute((const)); extern float __fabsf(float __x) throw() __attribute((const)); 
# 165
extern float floorf(float __x) throw() __attribute((const)); extern float __floorf(float __x) throw() __attribute((const)); 
# 168
extern float fmodf(float __x, float __y) throw(); extern float __fmodf(float __x, float __y) throw(); 
# 177 "/usr/include/bits/mathcalls.h" 3
extern int isinff(float __value) throw() __attribute((const)); 
# 182
extern int finitef(float __value) throw() __attribute((const)); 
# 185
extern float dremf(float __x, float __y) throw(); extern float __dremf(float __x, float __y) throw(); 
# 189
extern float significandf(float __x) throw(); extern float __significandf(float __x) throw(); 
# 196
extern float copysignf(float __x, float __y) throw() __attribute((const)); extern float __copysignf(float __x, float __y) throw() __attribute((const)); 
# 201
extern float nanf(const char * __tagb) throw(); extern float __nanf(const char * __tagb) throw(); 
# 211 "/usr/include/bits/mathcalls.h" 3
extern int isnanf(float __value) throw() __attribute((const)); 
# 217
extern float j0f(float) throw(); extern float __j0f(float) throw(); 
# 218
extern float j1f(float) throw(); extern float __j1f(float) throw(); 
# 219
extern float jnf(int, float) throw(); extern float __jnf(int, float) throw(); 
# 220
extern float y0f(float) throw(); extern float __y0f(float) throw(); 
# 221
extern float y1f(float) throw(); extern float __y1f(float) throw(); 
# 222
extern float ynf(int, float) throw(); extern float __ynf(int, float) throw(); 
# 228
extern float erff(float) throw(); extern float __erff(float) throw(); 
# 229
extern float erfcf(float) throw(); extern float __erfcf(float) throw(); 
# 230
extern float lgammaf(float) throw(); extern float __lgammaf(float) throw(); 
# 235
extern float tgammaf(float) throw(); extern float __tgammaf(float) throw(); 
# 241
extern float gammaf(float) throw(); extern float __gammaf(float) throw(); 
# 249
extern float lgammaf_r(float, int * __signgamp) throw(); extern float __lgammaf_r(float, int * __signgamp) throw(); 
# 256
extern float rintf(float __x) throw(); extern float __rintf(float __x) throw(); 
# 259
extern float nextafterf(float __x, float __y) throw(); extern float __nextafterf(float __x, float __y) throw(); 
# 261
extern float nexttowardf(float __x, long double __y) throw(); extern float __nexttowardf(float __x, long double __y) throw(); 
# 266
extern float nextdownf(float __x) throw(); extern float __nextdownf(float __x) throw(); 
# 268
extern float nextupf(float __x) throw(); extern float __nextupf(float __x) throw(); 
# 272
extern float remainderf(float __x, float __y) throw(); extern float __remainderf(float __x, float __y) throw(); 
# 276
extern float scalbnf(float __x, int __n) throw(); extern float __scalbnf(float __x, int __n) throw(); 
# 280
extern int ilogbf(float __x) throw(); extern int __ilogbf(float __x) throw(); 
# 285
extern long llogbf(float __x) throw(); extern long __llogbf(float __x) throw(); 
# 290
extern float scalblnf(float __x, long __n) throw(); extern float __scalblnf(float __x, long __n) throw(); 
# 294
extern float nearbyintf(float __x) throw(); extern float __nearbyintf(float __x) throw(); 
# 298
extern float roundf(float __x) throw() __attribute((const)); extern float __roundf(float __x) throw() __attribute((const)); 
# 302
extern float truncf(float __x) throw() __attribute((const)); extern float __truncf(float __x) throw() __attribute((const)); 
# 307
extern float remquof(float __x, float __y, int * __quo) throw(); extern float __remquof(float __x, float __y, int * __quo) throw(); 
# 314
extern long lrintf(float __x) throw(); extern long __lrintf(float __x) throw(); 
# 316
__extension__ extern long long llrintf(float __x) throw(); extern long long __llrintf(float __x) throw(); 
# 320
extern long lroundf(float __x) throw(); extern long __lroundf(float __x) throw(); 
# 322
__extension__ extern long long llroundf(float __x) throw(); extern long long __llroundf(float __x) throw(); 
# 326
extern float fdimf(float __x, float __y) throw(); extern float __fdimf(float __x, float __y) throw(); 
# 329
extern float fmaxf(float __x, float __y) throw() __attribute((const)); extern float __fmaxf(float __x, float __y) throw() __attribute((const)); 
# 332
extern float fminf(float __x, float __y) throw() __attribute((const)); extern float __fminf(float __x, float __y) throw() __attribute((const)); 
# 335
extern float fmaf(float __x, float __y, float __z) throw(); extern float __fmaf(float __x, float __y, float __z) throw(); 
# 340
extern float roundevenf(float __x) throw() __attribute((const)); extern float __roundevenf(float __x) throw() __attribute((const)); 
# 344
extern __intmax_t fromfpf(float __x, int __round, unsigned __width) throw(); extern __intmax_t __fromfpf(float __x, int __round, unsigned __width) throw(); 
# 349
extern __uintmax_t ufromfpf(float __x, int __round, unsigned __width) throw(); extern __uintmax_t __ufromfpf(float __x, int __round, unsigned __width) throw(); 
# 355
extern __intmax_t fromfpxf(float __x, int __round, unsigned __width) throw(); extern __intmax_t __fromfpxf(float __x, int __round, unsigned __width) throw(); 
# 361
extern __uintmax_t ufromfpxf(float __x, int __round, unsigned __width) throw(); extern __uintmax_t __ufromfpxf(float __x, int __round, unsigned __width) throw(); 
# 365
extern float fmaxmagf(float __x, float __y) throw() __attribute((const)); extern float __fmaxmagf(float __x, float __y) throw() __attribute((const)); 
# 368
extern float fminmagf(float __x, float __y) throw() __attribute((const)); extern float __fminmagf(float __x, float __y) throw() __attribute((const)); 
# 371
extern int totalorderf(float __x, float __y) throw()
# 372
 __attribute((const)); 
# 375
extern int totalordermagf(float __x, float __y) throw()
# 376
 __attribute((const)); 
# 379
extern int canonicalizef(float * __cx, const float * __x) throw(); 
# 382
extern float getpayloadf(const float * __x) throw(); extern float __getpayloadf(const float * __x) throw(); 
# 385
extern int setpayloadf(float * __x, float __payload) throw(); 
# 388
extern int setpayloadsigf(float * __x, float __payload) throw(); 
# 396
extern float scalbf(float __x, float __n) throw(); extern float __scalbf(float __x, float __n) throw(); 
# 21 "/usr/include/bits/mathcalls-helper-functions.h" 3
extern int __fpclassifyl(long double __value) throw()
# 22
 __attribute((const)); 
# 25
extern int __signbitl(long double __value) throw()
# 26
 __attribute((const)); 
# 30
extern int __isinfl(long double __value) throw() __attribute((const)); 
# 33
extern int __finitel(long double __value) throw() __attribute((const)); 
# 36
extern int __isnanl(long double __value) throw() __attribute((const)); 
# 39
extern int __iseqsigl(long double __x, long double __y) throw(); 
# 42
extern int __issignalingl(long double __value) throw()
# 43
 __attribute((const)); 
# 53 "/usr/include/bits/mathcalls.h" 3
extern long double acosl(long double __x) throw(); extern long double __acosl(long double __x) throw(); 
# 55
extern long double asinl(long double __x) throw(); extern long double __asinl(long double __x) throw(); 
# 57
extern long double atanl(long double __x) throw(); extern long double __atanl(long double __x) throw(); 
# 59
extern long double atan2l(long double __y, long double __x) throw(); extern long double __atan2l(long double __y, long double __x) throw(); 
# 62
extern long double cosl(long double __x) throw(); extern long double __cosl(long double __x) throw(); 
# 64
extern long double sinl(long double __x) throw(); extern long double __sinl(long double __x) throw(); 
# 66
extern long double tanl(long double __x) throw(); extern long double __tanl(long double __x) throw(); 
# 71
extern long double coshl(long double __x) throw(); extern long double __coshl(long double __x) throw(); 
# 73
extern long double sinhl(long double __x) throw(); extern long double __sinhl(long double __x) throw(); 
# 75
extern long double tanhl(long double __x) throw(); extern long double __tanhl(long double __x) throw(); 
# 79
extern void sincosl(long double __x, long double * __sinx, long double * __cosx) throw(); extern void __sincosl(long double __x, long double * __sinx, long double * __cosx) throw(); 
# 85
extern long double acoshl(long double __x) throw(); extern long double __acoshl(long double __x) throw(); 
# 87
extern long double asinhl(long double __x) throw(); extern long double __asinhl(long double __x) throw(); 
# 89
extern long double atanhl(long double __x) throw(); extern long double __atanhl(long double __x) throw(); 
# 95
extern long double expl(long double __x) throw(); extern long double __expl(long double __x) throw(); 
# 98
extern long double frexpl(long double __x, int * __exponent) throw(); extern long double __frexpl(long double __x, int * __exponent) throw(); 
# 101
extern long double ldexpl(long double __x, int __exponent) throw(); extern long double __ldexpl(long double __x, int __exponent) throw(); 
# 104
extern long double logl(long double __x) throw(); extern long double __logl(long double __x) throw(); 
# 107
extern long double log10l(long double __x) throw(); extern long double __log10l(long double __x) throw(); 
# 110
extern long double modfl(long double __x, long double * __iptr) throw(); extern long double __modfl(long double __x, long double * __iptr) throw() __attribute((__nonnull__(2))); 
# 114
extern long double exp10l(long double __x) throw(); extern long double __exp10l(long double __x) throw(); 
# 119
extern long double expm1l(long double __x) throw(); extern long double __expm1l(long double __x) throw(); 
# 122
extern long double log1pl(long double __x) throw(); extern long double __log1pl(long double __x) throw(); 
# 125
extern long double logbl(long double __x) throw(); extern long double __logbl(long double __x) throw(); 
# 130
extern long double exp2l(long double __x) throw(); extern long double __exp2l(long double __x) throw(); 
# 133
extern long double log2l(long double __x) throw(); extern long double __log2l(long double __x) throw(); 
# 140
extern long double powl(long double __x, long double __y) throw(); extern long double __powl(long double __x, long double __y) throw(); 
# 143
extern long double sqrtl(long double __x) throw(); extern long double __sqrtl(long double __x) throw(); 
# 147
extern long double hypotl(long double __x, long double __y) throw(); extern long double __hypotl(long double __x, long double __y) throw(); 
# 152
extern long double cbrtl(long double __x) throw(); extern long double __cbrtl(long double __x) throw(); 
# 159
extern long double ceill(long double __x) throw() __attribute((const)); extern long double __ceill(long double __x) throw() __attribute((const)); 
# 162
extern long double fabsl(long double __x) throw() __attribute((const)); extern long double __fabsl(long double __x) throw() __attribute((const)); 
# 165
extern long double floorl(long double __x) throw() __attribute((const)); extern long double __floorl(long double __x) throw() __attribute((const)); 
# 168
extern long double fmodl(long double __x, long double __y) throw(); extern long double __fmodl(long double __x, long double __y) throw(); 
# 177 "/usr/include/bits/mathcalls.h" 3
extern int isinfl(long double __value) throw() __attribute((const)); 
# 182
extern int finitel(long double __value) throw() __attribute((const)); 
# 185
extern long double dreml(long double __x, long double __y) throw(); extern long double __dreml(long double __x, long double __y) throw(); 
# 189
extern long double significandl(long double __x) throw(); extern long double __significandl(long double __x) throw(); 
# 196
extern long double copysignl(long double __x, long double __y) throw() __attribute((const)); extern long double __copysignl(long double __x, long double __y) throw() __attribute((const)); 
# 201
extern long double nanl(const char * __tagb) throw(); extern long double __nanl(const char * __tagb) throw(); 
# 211 "/usr/include/bits/mathcalls.h" 3
extern int isnanl(long double __value) throw() __attribute((const)); 
# 217
extern long double j0l(long double) throw(); extern long double __j0l(long double) throw(); 
# 218
extern long double j1l(long double) throw(); extern long double __j1l(long double) throw(); 
# 219
extern long double jnl(int, long double) throw(); extern long double __jnl(int, long double) throw(); 
# 220
extern long double y0l(long double) throw(); extern long double __y0l(long double) throw(); 
# 221
extern long double y1l(long double) throw(); extern long double __y1l(long double) throw(); 
# 222
extern long double ynl(int, long double) throw(); extern long double __ynl(int, long double) throw(); 
# 228
extern long double erfl(long double) throw(); extern long double __erfl(long double) throw(); 
# 229
extern long double erfcl(long double) throw(); extern long double __erfcl(long double) throw(); 
# 230
extern long double lgammal(long double) throw(); extern long double __lgammal(long double) throw(); 
# 235
extern long double tgammal(long double) throw(); extern long double __tgammal(long double) throw(); 
# 241
extern long double gammal(long double) throw(); extern long double __gammal(long double) throw(); 
# 249
extern long double lgammal_r(long double, int * __signgamp) throw(); extern long double __lgammal_r(long double, int * __signgamp) throw(); 
# 256
extern long double rintl(long double __x) throw(); extern long double __rintl(long double __x) throw(); 
# 259
extern long double nextafterl(long double __x, long double __y) throw(); extern long double __nextafterl(long double __x, long double __y) throw(); 
# 261
extern long double nexttowardl(long double __x, long double __y) throw(); extern long double __nexttowardl(long double __x, long double __y) throw(); 
# 266
extern long double nextdownl(long double __x) throw(); extern long double __nextdownl(long double __x) throw(); 
# 268
extern long double nextupl(long double __x) throw(); extern long double __nextupl(long double __x) throw(); 
# 272
extern long double remainderl(long double __x, long double __y) throw(); extern long double __remainderl(long double __x, long double __y) throw(); 
# 276
extern long double scalbnl(long double __x, int __n) throw(); extern long double __scalbnl(long double __x, int __n) throw(); 
# 280
extern int ilogbl(long double __x) throw(); extern int __ilogbl(long double __x) throw(); 
# 285
extern long llogbl(long double __x) throw(); extern long __llogbl(long double __x) throw(); 
# 290
extern long double scalblnl(long double __x, long __n) throw(); extern long double __scalblnl(long double __x, long __n) throw(); 
# 294
extern long double nearbyintl(long double __x) throw(); extern long double __nearbyintl(long double __x) throw(); 
# 298
extern long double roundl(long double __x) throw() __attribute((const)); extern long double __roundl(long double __x) throw() __attribute((const)); 
# 302
extern long double truncl(long double __x) throw() __attribute((const)); extern long double __truncl(long double __x) throw() __attribute((const)); 
# 307
extern long double remquol(long double __x, long double __y, int * __quo) throw(); extern long double __remquol(long double __x, long double __y, int * __quo) throw(); 
# 314
extern long lrintl(long double __x) throw(); extern long __lrintl(long double __x) throw(); 
# 316
__extension__ extern long long llrintl(long double __x) throw(); extern long long __llrintl(long double __x) throw(); 
# 320
extern long lroundl(long double __x) throw(); extern long __lroundl(long double __x) throw(); 
# 322
__extension__ extern long long llroundl(long double __x) throw(); extern long long __llroundl(long double __x) throw(); 
# 326
extern long double fdiml(long double __x, long double __y) throw(); extern long double __fdiml(long double __x, long double __y) throw(); 
# 329
extern long double fmaxl(long double __x, long double __y) throw() __attribute((const)); extern long double __fmaxl(long double __x, long double __y) throw() __attribute((const)); 
# 332
extern long double fminl(long double __x, long double __y) throw() __attribute((const)); extern long double __fminl(long double __x, long double __y) throw() __attribute((const)); 
# 335
extern long double fmal(long double __x, long double __y, long double __z) throw(); extern long double __fmal(long double __x, long double __y, long double __z) throw(); 
# 340
extern long double roundevenl(long double __x) throw() __attribute((const)); extern long double __roundevenl(long double __x) throw() __attribute((const)); 
# 344
extern __intmax_t fromfpl(long double __x, int __round, unsigned __width) throw(); extern __intmax_t __fromfpl(long double __x, int __round, unsigned __width) throw(); 
# 349
extern __uintmax_t ufromfpl(long double __x, int __round, unsigned __width) throw(); extern __uintmax_t __ufromfpl(long double __x, int __round, unsigned __width) throw(); 
# 355
extern __intmax_t fromfpxl(long double __x, int __round, unsigned __width) throw(); extern __intmax_t __fromfpxl(long double __x, int __round, unsigned __width) throw(); 
# 361
extern __uintmax_t ufromfpxl(long double __x, int __round, unsigned __width) throw(); extern __uintmax_t __ufromfpxl(long double __x, int __round, unsigned __width) throw(); 
# 365
extern long double fmaxmagl(long double __x, long double __y) throw() __attribute((const)); extern long double __fmaxmagl(long double __x, long double __y) throw() __attribute((const)); 
# 368
extern long double fminmagl(long double __x, long double __y) throw() __attribute((const)); extern long double __fminmagl(long double __x, long double __y) throw() __attribute((const)); 
# 371
extern int totalorderl(long double __x, long double __y) throw()
# 372
 __attribute((const)); 
# 375
extern int totalordermagl(long double __x, long double __y) throw()
# 376
 __attribute((const)); 
# 379
extern int canonicalizel(long double * __cx, const long double * __x) throw(); 
# 382
extern long double getpayloadl(const long double * __x) throw(); extern long double __getpayloadl(const long double * __x) throw(); 
# 385
extern int setpayloadl(long double * __x, long double __payload) throw(); 
# 388
extern int setpayloadsigl(long double * __x, long double __payload) throw(); 
# 396
extern long double scalbl(long double __x, long double __n) throw(); extern long double __scalbl(long double __x, long double __n) throw(); 
# 53 "/usr/include/bits/mathcalls.h" 3
extern _Float32 acosf32(_Float32 __x) throw(); extern _Float32 __acosf32(_Float32 __x) throw(); 
# 55
extern _Float32 asinf32(_Float32 __x) throw(); extern _Float32 __asinf32(_Float32 __x) throw(); 
# 57
extern _Float32 atanf32(_Float32 __x) throw(); extern _Float32 __atanf32(_Float32 __x) throw(); 
# 59
extern _Float32 atan2f32(_Float32 __y, _Float32 __x) throw(); extern _Float32 __atan2f32(_Float32 __y, _Float32 __x) throw(); 
# 62
extern _Float32 cosf32(_Float32 __x) throw(); extern _Float32 __cosf32(_Float32 __x) throw(); 
# 64
extern _Float32 sinf32(_Float32 __x) throw(); extern _Float32 __sinf32(_Float32 __x) throw(); 
# 66
extern _Float32 tanf32(_Float32 __x) throw(); extern _Float32 __tanf32(_Float32 __x) throw(); 
# 71
extern _Float32 coshf32(_Float32 __x) throw(); extern _Float32 __coshf32(_Float32 __x) throw(); 
# 73
extern _Float32 sinhf32(_Float32 __x) throw(); extern _Float32 __sinhf32(_Float32 __x) throw(); 
# 75
extern _Float32 tanhf32(_Float32 __x) throw(); extern _Float32 __tanhf32(_Float32 __x) throw(); 
# 79
extern void sincosf32(_Float32 __x, _Float32 * __sinx, _Float32 * __cosx) throw(); extern void __sincosf32(_Float32 __x, _Float32 * __sinx, _Float32 * __cosx) throw(); 
# 85
extern _Float32 acoshf32(_Float32 __x) throw(); extern _Float32 __acoshf32(_Float32 __x) throw(); 
# 87
extern _Float32 asinhf32(_Float32 __x) throw(); extern _Float32 __asinhf32(_Float32 __x) throw(); 
# 89
extern _Float32 atanhf32(_Float32 __x) throw(); extern _Float32 __atanhf32(_Float32 __x) throw(); 
# 95
extern _Float32 expf32(_Float32 __x) throw(); extern _Float32 __expf32(_Float32 __x) throw(); 
# 98
extern _Float32 frexpf32(_Float32 __x, int * __exponent) throw(); extern _Float32 __frexpf32(_Float32 __x, int * __exponent) throw(); 
# 101
extern _Float32 ldexpf32(_Float32 __x, int __exponent) throw(); extern _Float32 __ldexpf32(_Float32 __x, int __exponent) throw(); 
# 104
extern _Float32 logf32(_Float32 __x) throw(); extern _Float32 __logf32(_Float32 __x) throw(); 
# 107
extern _Float32 log10f32(_Float32 __x) throw(); extern _Float32 __log10f32(_Float32 __x) throw(); 
# 110
extern _Float32 modff32(_Float32 __x, _Float32 * __iptr) throw(); extern _Float32 __modff32(_Float32 __x, _Float32 * __iptr) throw() __attribute((__nonnull__(2))); 
# 114
extern _Float32 exp10f32(_Float32 __x) throw(); extern _Float32 __exp10f32(_Float32 __x) throw(); 
# 119
extern _Float32 expm1f32(_Float32 __x) throw(); extern _Float32 __expm1f32(_Float32 __x) throw(); 
# 122
extern _Float32 log1pf32(_Float32 __x) throw(); extern _Float32 __log1pf32(_Float32 __x) throw(); 
# 125
extern _Float32 logbf32(_Float32 __x) throw(); extern _Float32 __logbf32(_Float32 __x) throw(); 
# 130
extern _Float32 exp2f32(_Float32 __x) throw(); extern _Float32 __exp2f32(_Float32 __x) throw(); 
# 133
extern _Float32 log2f32(_Float32 __x) throw(); extern _Float32 __log2f32(_Float32 __x) throw(); 
# 140
extern _Float32 powf32(_Float32 __x, _Float32 __y) throw(); extern _Float32 __powf32(_Float32 __x, _Float32 __y) throw(); 
# 143
extern _Float32 sqrtf32(_Float32 __x) throw(); extern _Float32 __sqrtf32(_Float32 __x) throw(); 
# 147
extern _Float32 hypotf32(_Float32 __x, _Float32 __y) throw(); extern _Float32 __hypotf32(_Float32 __x, _Float32 __y) throw(); 
# 152
extern _Float32 cbrtf32(_Float32 __x) throw(); extern _Float32 __cbrtf32(_Float32 __x) throw(); 
# 159
extern _Float32 ceilf32(_Float32 __x) throw() __attribute((const)); extern _Float32 __ceilf32(_Float32 __x) throw() __attribute((const)); 
# 162
extern _Float32 fabsf32(_Float32 __x) throw() __attribute((const)); extern _Float32 __fabsf32(_Float32 __x) throw() __attribute((const)); 
# 165
extern _Float32 floorf32(_Float32 __x) throw() __attribute((const)); extern _Float32 __floorf32(_Float32 __x) throw() __attribute((const)); 
# 168
extern _Float32 fmodf32(_Float32 __x, _Float32 __y) throw(); extern _Float32 __fmodf32(_Float32 __x, _Float32 __y) throw(); 
# 196 "/usr/include/bits/mathcalls.h" 3
extern _Float32 copysignf32(_Float32 __x, _Float32 __y) throw() __attribute((const)); extern _Float32 __copysignf32(_Float32 __x, _Float32 __y) throw() __attribute((const)); 
# 201
extern _Float32 nanf32(const char * __tagb) throw(); extern _Float32 __nanf32(const char * __tagb) throw(); 
# 217 "/usr/include/bits/mathcalls.h" 3
extern _Float32 j0f32(_Float32) throw(); extern _Float32 __j0f32(_Float32) throw(); 
# 218
extern _Float32 j1f32(_Float32) throw(); extern _Float32 __j1f32(_Float32) throw(); 
# 219
extern _Float32 jnf32(int, _Float32) throw(); extern _Float32 __jnf32(int, _Float32) throw(); 
# 220
extern _Float32 y0f32(_Float32) throw(); extern _Float32 __y0f32(_Float32) throw(); 
# 221
extern _Float32 y1f32(_Float32) throw(); extern _Float32 __y1f32(_Float32) throw(); 
# 222
extern _Float32 ynf32(int, _Float32) throw(); extern _Float32 __ynf32(int, _Float32) throw(); 
# 228
extern _Float32 erff32(_Float32) throw(); extern _Float32 __erff32(_Float32) throw(); 
# 229
extern _Float32 erfcf32(_Float32) throw(); extern _Float32 __erfcf32(_Float32) throw(); 
# 230
extern _Float32 lgammaf32(_Float32) throw(); extern _Float32 __lgammaf32(_Float32) throw(); 
# 235
extern _Float32 tgammaf32(_Float32) throw(); extern _Float32 __tgammaf32(_Float32) throw(); 
# 249 "/usr/include/bits/mathcalls.h" 3
extern _Float32 lgammaf32_r(_Float32, int * __signgamp) throw(); extern _Float32 __lgammaf32_r(_Float32, int * __signgamp) throw(); 
# 256
extern _Float32 rintf32(_Float32 __x) throw(); extern _Float32 __rintf32(_Float32 __x) throw(); 
# 259
extern _Float32 nextafterf32(_Float32 __x, _Float32 __y) throw(); extern _Float32 __nextafterf32(_Float32 __x, _Float32 __y) throw(); 
# 266
extern _Float32 nextdownf32(_Float32 __x) throw(); extern _Float32 __nextdownf32(_Float32 __x) throw(); 
# 268
extern _Float32 nextupf32(_Float32 __x) throw(); extern _Float32 __nextupf32(_Float32 __x) throw(); 
# 272
extern _Float32 remainderf32(_Float32 __x, _Float32 __y) throw(); extern _Float32 __remainderf32(_Float32 __x, _Float32 __y) throw(); 
# 276
extern _Float32 scalbnf32(_Float32 __x, int __n) throw(); extern _Float32 __scalbnf32(_Float32 __x, int __n) throw(); 
# 280
extern int ilogbf32(_Float32 __x) throw(); extern int __ilogbf32(_Float32 __x) throw(); 
# 285
extern long llogbf32(_Float32 __x) throw(); extern long __llogbf32(_Float32 __x) throw(); 
# 290
extern _Float32 scalblnf32(_Float32 __x, long __n) throw(); extern _Float32 __scalblnf32(_Float32 __x, long __n) throw(); 
# 294
extern _Float32 nearbyintf32(_Float32 __x) throw(); extern _Float32 __nearbyintf32(_Float32 __x) throw(); 
# 298
extern _Float32 roundf32(_Float32 __x) throw() __attribute((const)); extern _Float32 __roundf32(_Float32 __x) throw() __attribute((const)); 
# 302
extern _Float32 truncf32(_Float32 __x) throw() __attribute((const)); extern _Float32 __truncf32(_Float32 __x) throw() __attribute((const)); 
# 307
extern _Float32 remquof32(_Float32 __x, _Float32 __y, int * __quo) throw(); extern _Float32 __remquof32(_Float32 __x, _Float32 __y, int * __quo) throw(); 
# 314
extern long lrintf32(_Float32 __x) throw(); extern long __lrintf32(_Float32 __x) throw(); 
# 316
__extension__ extern long long llrintf32(_Float32 __x) throw(); extern long long __llrintf32(_Float32 __x) throw(); 
# 320
extern long lroundf32(_Float32 __x) throw(); extern long __lroundf32(_Float32 __x) throw(); 
# 322
__extension__ extern long long llroundf32(_Float32 __x) throw(); extern long long __llroundf32(_Float32 __x) throw(); 
# 326
extern _Float32 fdimf32(_Float32 __x, _Float32 __y) throw(); extern _Float32 __fdimf32(_Float32 __x, _Float32 __y) throw(); 
# 329
extern _Float32 fmaxf32(_Float32 __x, _Float32 __y) throw() __attribute((const)); extern _Float32 __fmaxf32(_Float32 __x, _Float32 __y) throw() __attribute((const)); 
# 332
extern _Float32 fminf32(_Float32 __x, _Float32 __y) throw() __attribute((const)); extern _Float32 __fminf32(_Float32 __x, _Float32 __y) throw() __attribute((const)); 
# 335
extern _Float32 fmaf32(_Float32 __x, _Float32 __y, _Float32 __z) throw(); extern _Float32 __fmaf32(_Float32 __x, _Float32 __y, _Float32 __z) throw(); 
# 340
extern _Float32 roundevenf32(_Float32 __x) throw() __attribute((const)); extern _Float32 __roundevenf32(_Float32 __x) throw() __attribute((const)); 
# 344
extern __intmax_t fromfpf32(_Float32 __x, int __round, unsigned __width) throw(); extern __intmax_t __fromfpf32(_Float32 __x, int __round, unsigned __width) throw(); 
# 349
extern __uintmax_t ufromfpf32(_Float32 __x, int __round, unsigned __width) throw(); extern __uintmax_t __ufromfpf32(_Float32 __x, int __round, unsigned __width) throw(); 
# 355
extern __intmax_t fromfpxf32(_Float32 __x, int __round, unsigned __width) throw(); extern __intmax_t __fromfpxf32(_Float32 __x, int __round, unsigned __width) throw(); 
# 361
extern __uintmax_t ufromfpxf32(_Float32 __x, int __round, unsigned __width) throw(); extern __uintmax_t __ufromfpxf32(_Float32 __x, int __round, unsigned __width) throw(); 
# 365
extern _Float32 fmaxmagf32(_Float32 __x, _Float32 __y) throw() __attribute((const)); extern _Float32 __fmaxmagf32(_Float32 __x, _Float32 __y) throw() __attribute((const)); 
# 368
extern _Float32 fminmagf32(_Float32 __x, _Float32 __y) throw() __attribute((const)); extern _Float32 __fminmagf32(_Float32 __x, _Float32 __y) throw() __attribute((const)); 
# 371
extern int totalorderf32(_Float32 __x, _Float32 __y) throw()
# 372
 __attribute((const)); 
# 375
extern int totalordermagf32(_Float32 __x, _Float32 __y) throw()
# 376
 __attribute((const)); 
# 379
extern int canonicalizef32(_Float32 * __cx, const _Float32 * __x) throw(); 
# 382
extern _Float32 getpayloadf32(const _Float32 * __x) throw(); extern _Float32 __getpayloadf32(const _Float32 * __x) throw(); 
# 385
extern int setpayloadf32(_Float32 * __x, _Float32 __payload) throw(); 
# 388
extern int setpayloadsigf32(_Float32 * __x, _Float32 __payload) throw(); 
# 53 "/usr/include/bits/mathcalls.h" 3
extern _Float64 acosf64(_Float64 __x) throw(); extern _Float64 __acosf64(_Float64 __x) throw(); 
# 55
extern _Float64 asinf64(_Float64 __x) throw(); extern _Float64 __asinf64(_Float64 __x) throw(); 
# 57
extern _Float64 atanf64(_Float64 __x) throw(); extern _Float64 __atanf64(_Float64 __x) throw(); 
# 59
extern _Float64 atan2f64(_Float64 __y, _Float64 __x) throw(); extern _Float64 __atan2f64(_Float64 __y, _Float64 __x) throw(); 
# 62
extern _Float64 cosf64(_Float64 __x) throw(); extern _Float64 __cosf64(_Float64 __x) throw(); 
# 64
extern _Float64 sinf64(_Float64 __x) throw(); extern _Float64 __sinf64(_Float64 __x) throw(); 
# 66
extern _Float64 tanf64(_Float64 __x) throw(); extern _Float64 __tanf64(_Float64 __x) throw(); 
# 71
extern _Float64 coshf64(_Float64 __x) throw(); extern _Float64 __coshf64(_Float64 __x) throw(); 
# 73
extern _Float64 sinhf64(_Float64 __x) throw(); extern _Float64 __sinhf64(_Float64 __x) throw(); 
# 75
extern _Float64 tanhf64(_Float64 __x) throw(); extern _Float64 __tanhf64(_Float64 __x) throw(); 
# 79
extern void sincosf64(_Float64 __x, _Float64 * __sinx, _Float64 * __cosx) throw(); extern void __sincosf64(_Float64 __x, _Float64 * __sinx, _Float64 * __cosx) throw(); 
# 85
extern _Float64 acoshf64(_Float64 __x) throw(); extern _Float64 __acoshf64(_Float64 __x) throw(); 
# 87
extern _Float64 asinhf64(_Float64 __x) throw(); extern _Float64 __asinhf64(_Float64 __x) throw(); 
# 89
extern _Float64 atanhf64(_Float64 __x) throw(); extern _Float64 __atanhf64(_Float64 __x) throw(); 
# 95
extern _Float64 expf64(_Float64 __x) throw(); extern _Float64 __expf64(_Float64 __x) throw(); 
# 98
extern _Float64 frexpf64(_Float64 __x, int * __exponent) throw(); extern _Float64 __frexpf64(_Float64 __x, int * __exponent) throw(); 
# 101
extern _Float64 ldexpf64(_Float64 __x, int __exponent) throw(); extern _Float64 __ldexpf64(_Float64 __x, int __exponent) throw(); 
# 104
extern _Float64 logf64(_Float64 __x) throw(); extern _Float64 __logf64(_Float64 __x) throw(); 
# 107
extern _Float64 log10f64(_Float64 __x) throw(); extern _Float64 __log10f64(_Float64 __x) throw(); 
# 110
extern _Float64 modff64(_Float64 __x, _Float64 * __iptr) throw(); extern _Float64 __modff64(_Float64 __x, _Float64 * __iptr) throw() __attribute((__nonnull__(2))); 
# 114
extern _Float64 exp10f64(_Float64 __x) throw(); extern _Float64 __exp10f64(_Float64 __x) throw(); 
# 119
extern _Float64 expm1f64(_Float64 __x) throw(); extern _Float64 __expm1f64(_Float64 __x) throw(); 
# 122
extern _Float64 log1pf64(_Float64 __x) throw(); extern _Float64 __log1pf64(_Float64 __x) throw(); 
# 125
extern _Float64 logbf64(_Float64 __x) throw(); extern _Float64 __logbf64(_Float64 __x) throw(); 
# 130
extern _Float64 exp2f64(_Float64 __x) throw(); extern _Float64 __exp2f64(_Float64 __x) throw(); 
# 133
extern _Float64 log2f64(_Float64 __x) throw(); extern _Float64 __log2f64(_Float64 __x) throw(); 
# 140
extern _Float64 powf64(_Float64 __x, _Float64 __y) throw(); extern _Float64 __powf64(_Float64 __x, _Float64 __y) throw(); 
# 143
extern _Float64 sqrtf64(_Float64 __x) throw(); extern _Float64 __sqrtf64(_Float64 __x) throw(); 
# 147
extern _Float64 hypotf64(_Float64 __x, _Float64 __y) throw(); extern _Float64 __hypotf64(_Float64 __x, _Float64 __y) throw(); 
# 152
extern _Float64 cbrtf64(_Float64 __x) throw(); extern _Float64 __cbrtf64(_Float64 __x) throw(); 
# 159
extern _Float64 ceilf64(_Float64 __x) throw() __attribute((const)); extern _Float64 __ceilf64(_Float64 __x) throw() __attribute((const)); 
# 162
extern _Float64 fabsf64(_Float64 __x) throw() __attribute((const)); extern _Float64 __fabsf64(_Float64 __x) throw() __attribute((const)); 
# 165
extern _Float64 floorf64(_Float64 __x) throw() __attribute((const)); extern _Float64 __floorf64(_Float64 __x) throw() __attribute((const)); 
# 168
extern _Float64 fmodf64(_Float64 __x, _Float64 __y) throw(); extern _Float64 __fmodf64(_Float64 __x, _Float64 __y) throw(); 
# 196 "/usr/include/bits/mathcalls.h" 3
extern _Float64 copysignf64(_Float64 __x, _Float64 __y) throw() __attribute((const)); extern _Float64 __copysignf64(_Float64 __x, _Float64 __y) throw() __attribute((const)); 
# 201
extern _Float64 nanf64(const char * __tagb) throw(); extern _Float64 __nanf64(const char * __tagb) throw(); 
# 217 "/usr/include/bits/mathcalls.h" 3
extern _Float64 j0f64(_Float64) throw(); extern _Float64 __j0f64(_Float64) throw(); 
# 218
extern _Float64 j1f64(_Float64) throw(); extern _Float64 __j1f64(_Float64) throw(); 
# 219
extern _Float64 jnf64(int, _Float64) throw(); extern _Float64 __jnf64(int, _Float64) throw(); 
# 220
extern _Float64 y0f64(_Float64) throw(); extern _Float64 __y0f64(_Float64) throw(); 
# 221
extern _Float64 y1f64(_Float64) throw(); extern _Float64 __y1f64(_Float64) throw(); 
# 222
extern _Float64 ynf64(int, _Float64) throw(); extern _Float64 __ynf64(int, _Float64) throw(); 
# 228
extern _Float64 erff64(_Float64) throw(); extern _Float64 __erff64(_Float64) throw(); 
# 229
extern _Float64 erfcf64(_Float64) throw(); extern _Float64 __erfcf64(_Float64) throw(); 
# 230
extern _Float64 lgammaf64(_Float64) throw(); extern _Float64 __lgammaf64(_Float64) throw(); 
# 235
extern _Float64 tgammaf64(_Float64) throw(); extern _Float64 __tgammaf64(_Float64) throw(); 
# 249 "/usr/include/bits/mathcalls.h" 3
extern _Float64 lgammaf64_r(_Float64, int * __signgamp) throw(); extern _Float64 __lgammaf64_r(_Float64, int * __signgamp) throw(); 
# 256
extern _Float64 rintf64(_Float64 __x) throw(); extern _Float64 __rintf64(_Float64 __x) throw(); 
# 259
extern _Float64 nextafterf64(_Float64 __x, _Float64 __y) throw(); extern _Float64 __nextafterf64(_Float64 __x, _Float64 __y) throw(); 
# 266
extern _Float64 nextdownf64(_Float64 __x) throw(); extern _Float64 __nextdownf64(_Float64 __x) throw(); 
# 268
extern _Float64 nextupf64(_Float64 __x) throw(); extern _Float64 __nextupf64(_Float64 __x) throw(); 
# 272
extern _Float64 remainderf64(_Float64 __x, _Float64 __y) throw(); extern _Float64 __remainderf64(_Float64 __x, _Float64 __y) throw(); 
# 276
extern _Float64 scalbnf64(_Float64 __x, int __n) throw(); extern _Float64 __scalbnf64(_Float64 __x, int __n) throw(); 
# 280
extern int ilogbf64(_Float64 __x) throw(); extern int __ilogbf64(_Float64 __x) throw(); 
# 285
extern long llogbf64(_Float64 __x) throw(); extern long __llogbf64(_Float64 __x) throw(); 
# 290
extern _Float64 scalblnf64(_Float64 __x, long __n) throw(); extern _Float64 __scalblnf64(_Float64 __x, long __n) throw(); 
# 294
extern _Float64 nearbyintf64(_Float64 __x) throw(); extern _Float64 __nearbyintf64(_Float64 __x) throw(); 
# 298
extern _Float64 roundf64(_Float64 __x) throw() __attribute((const)); extern _Float64 __roundf64(_Float64 __x) throw() __attribute((const)); 
# 302
extern _Float64 truncf64(_Float64 __x) throw() __attribute((const)); extern _Float64 __truncf64(_Float64 __x) throw() __attribute((const)); 
# 307
extern _Float64 remquof64(_Float64 __x, _Float64 __y, int * __quo) throw(); extern _Float64 __remquof64(_Float64 __x, _Float64 __y, int * __quo) throw(); 
# 314
extern long lrintf64(_Float64 __x) throw(); extern long __lrintf64(_Float64 __x) throw(); 
# 316
__extension__ extern long long llrintf64(_Float64 __x) throw(); extern long long __llrintf64(_Float64 __x) throw(); 
# 320
extern long lroundf64(_Float64 __x) throw(); extern long __lroundf64(_Float64 __x) throw(); 
# 322
__extension__ extern long long llroundf64(_Float64 __x) throw(); extern long long __llroundf64(_Float64 __x) throw(); 
# 326
extern _Float64 fdimf64(_Float64 __x, _Float64 __y) throw(); extern _Float64 __fdimf64(_Float64 __x, _Float64 __y) throw(); 
# 329
extern _Float64 fmaxf64(_Float64 __x, _Float64 __y) throw() __attribute((const)); extern _Float64 __fmaxf64(_Float64 __x, _Float64 __y) throw() __attribute((const)); 
# 332
extern _Float64 fminf64(_Float64 __x, _Float64 __y) throw() __attribute((const)); extern _Float64 __fminf64(_Float64 __x, _Float64 __y) throw() __attribute((const)); 
# 335
extern _Float64 fmaf64(_Float64 __x, _Float64 __y, _Float64 __z) throw(); extern _Float64 __fmaf64(_Float64 __x, _Float64 __y, _Float64 __z) throw(); 
# 340
extern _Float64 roundevenf64(_Float64 __x) throw() __attribute((const)); extern _Float64 __roundevenf64(_Float64 __x) throw() __attribute((const)); 
# 344
extern __intmax_t fromfpf64(_Float64 __x, int __round, unsigned __width) throw(); extern __intmax_t __fromfpf64(_Float64 __x, int __round, unsigned __width) throw(); 
# 349
extern __uintmax_t ufromfpf64(_Float64 __x, int __round, unsigned __width) throw(); extern __uintmax_t __ufromfpf64(_Float64 __x, int __round, unsigned __width) throw(); 
# 355
extern __intmax_t fromfpxf64(_Float64 __x, int __round, unsigned __width) throw(); extern __intmax_t __fromfpxf64(_Float64 __x, int __round, unsigned __width) throw(); 
# 361
extern __uintmax_t ufromfpxf64(_Float64 __x, int __round, unsigned __width) throw(); extern __uintmax_t __ufromfpxf64(_Float64 __x, int __round, unsigned __width) throw(); 
# 365
extern _Float64 fmaxmagf64(_Float64 __x, _Float64 __y) throw() __attribute((const)); extern _Float64 __fmaxmagf64(_Float64 __x, _Float64 __y) throw() __attribute((const)); 
# 368
extern _Float64 fminmagf64(_Float64 __x, _Float64 __y) throw() __attribute((const)); extern _Float64 __fminmagf64(_Float64 __x, _Float64 __y) throw() __attribute((const)); 
# 371
extern int totalorderf64(_Float64 __x, _Float64 __y) throw()
# 372
 __attribute((const)); 
# 375
extern int totalordermagf64(_Float64 __x, _Float64 __y) throw()
# 376
 __attribute((const)); 
# 379
extern int canonicalizef64(_Float64 * __cx, const _Float64 * __x) throw(); 
# 382
extern _Float64 getpayloadf64(const _Float64 * __x) throw(); extern _Float64 __getpayloadf64(const _Float64 * __x) throw(); 
# 385
extern int setpayloadf64(_Float64 * __x, _Float64 __payload) throw(); 
# 388
extern int setpayloadsigf64(_Float64 * __x, _Float64 __payload) throw(); 
# 21 "/usr/include/bits/mathcalls-helper-functions.h" 3
extern int __fpclassifyf128(_Float128 __value) throw()
# 22
 __attribute((const)); 
# 25
extern int __signbitf128(_Float128 __value) throw()
# 26
 __attribute((const)); 
# 30
extern int __isinff128(_Float128 __value) throw() __attribute((const)); 
# 33
extern int __finitef128(_Float128 __value) throw() __attribute((const)); 
# 36
extern int __isnanf128(_Float128 __value) throw() __attribute((const)); 
# 39
extern int __iseqsigf128(_Float128 __x, _Float128 __y) throw(); 
# 42
extern int __issignalingf128(_Float128 __value) throw()
# 43
 __attribute((const)); 
# 53 "/usr/include/bits/mathcalls.h" 3
extern _Float128 acosf128(_Float128 __x) throw(); extern _Float128 __acosf128(_Float128 __x) throw(); 
# 55
extern _Float128 asinf128(_Float128 __x) throw(); extern _Float128 __asinf128(_Float128 __x) throw(); 
# 57
extern _Float128 atanf128(_Float128 __x) throw(); extern _Float128 __atanf128(_Float128 __x) throw(); 
# 59
extern _Float128 atan2f128(_Float128 __y, _Float128 __x) throw(); extern _Float128 __atan2f128(_Float128 __y, _Float128 __x) throw(); 
# 62
extern _Float128 cosf128(_Float128 __x) throw(); extern _Float128 __cosf128(_Float128 __x) throw(); 
# 64
extern _Float128 sinf128(_Float128 __x) throw(); extern _Float128 __sinf128(_Float128 __x) throw(); 
# 66
extern _Float128 tanf128(_Float128 __x) throw(); extern _Float128 __tanf128(_Float128 __x) throw(); 
# 71
extern _Float128 coshf128(_Float128 __x) throw(); extern _Float128 __coshf128(_Float128 __x) throw(); 
# 73
extern _Float128 sinhf128(_Float128 __x) throw(); extern _Float128 __sinhf128(_Float128 __x) throw(); 
# 75
extern _Float128 tanhf128(_Float128 __x) throw(); extern _Float128 __tanhf128(_Float128 __x) throw(); 
# 79
extern void sincosf128(_Float128 __x, _Float128 * __sinx, _Float128 * __cosx) throw(); extern void __sincosf128(_Float128 __x, _Float128 * __sinx, _Float128 * __cosx) throw(); 
# 85
extern _Float128 acoshf128(_Float128 __x) throw(); extern _Float128 __acoshf128(_Float128 __x) throw(); 
# 87
extern _Float128 asinhf128(_Float128 __x) throw(); extern _Float128 __asinhf128(_Float128 __x) throw(); 
# 89
extern _Float128 atanhf128(_Float128 __x) throw(); extern _Float128 __atanhf128(_Float128 __x) throw(); 
# 95
extern _Float128 expf128(_Float128 __x) throw(); extern _Float128 __expf128(_Float128 __x) throw(); 
# 98
extern _Float128 frexpf128(_Float128 __x, int * __exponent) throw(); extern _Float128 __frexpf128(_Float128 __x, int * __exponent) throw(); 
# 101
extern _Float128 ldexpf128(_Float128 __x, int __exponent) throw(); extern _Float128 __ldexpf128(_Float128 __x, int __exponent) throw(); 
# 104
extern _Float128 logf128(_Float128 __x) throw(); extern _Float128 __logf128(_Float128 __x) throw(); 
# 107
extern _Float128 log10f128(_Float128 __x) throw(); extern _Float128 __log10f128(_Float128 __x) throw(); 
# 110
extern _Float128 modff128(_Float128 __x, _Float128 * __iptr) throw(); extern _Float128 __modff128(_Float128 __x, _Float128 * __iptr) throw() __attribute((__nonnull__(2))); 
# 114
extern _Float128 exp10f128(_Float128 __x) throw(); extern _Float128 __exp10f128(_Float128 __x) throw(); 
# 119
extern _Float128 expm1f128(_Float128 __x) throw(); extern _Float128 __expm1f128(_Float128 __x) throw(); 
# 122
extern _Float128 log1pf128(_Float128 __x) throw(); extern _Float128 __log1pf128(_Float128 __x) throw(); 
# 125
extern _Float128 logbf128(_Float128 __x) throw(); extern _Float128 __logbf128(_Float128 __x) throw(); 
# 130
extern _Float128 exp2f128(_Float128 __x) throw(); extern _Float128 __exp2f128(_Float128 __x) throw(); 
# 133
extern _Float128 log2f128(_Float128 __x) throw(); extern _Float128 __log2f128(_Float128 __x) throw(); 
# 140
extern _Float128 powf128(_Float128 __x, _Float128 __y) throw(); extern _Float128 __powf128(_Float128 __x, _Float128 __y) throw(); 
# 143
extern _Float128 sqrtf128(_Float128 __x) throw(); extern _Float128 __sqrtf128(_Float128 __x) throw(); 
# 147
extern _Float128 hypotf128(_Float128 __x, _Float128 __y) throw(); extern _Float128 __hypotf128(_Float128 __x, _Float128 __y) throw(); 
# 152
extern _Float128 cbrtf128(_Float128 __x) throw(); extern _Float128 __cbrtf128(_Float128 __x) throw(); 
# 159
extern _Float128 ceilf128(_Float128 __x) throw() __attribute((const)); extern _Float128 __ceilf128(_Float128 __x) throw() __attribute((const)); 
# 162
extern _Float128 fabsf128(_Float128 __x) throw() __attribute((const)); extern _Float128 __fabsf128(_Float128 __x) throw() __attribute((const)); 
# 165
extern _Float128 floorf128(_Float128 __x) throw() __attribute((const)); extern _Float128 __floorf128(_Float128 __x) throw() __attribute((const)); 
# 168
extern _Float128 fmodf128(_Float128 __x, _Float128 __y) throw(); extern _Float128 __fmodf128(_Float128 __x, _Float128 __y) throw(); 
# 196 "/usr/include/bits/mathcalls.h" 3
extern _Float128 copysignf128(_Float128 __x, _Float128 __y) throw() __attribute((const)); extern _Float128 __copysignf128(_Float128 __x, _Float128 __y) throw() __attribute((const)); 
# 201
extern _Float128 nanf128(const char * __tagb) throw(); extern _Float128 __nanf128(const char * __tagb) throw(); 
# 217 "/usr/include/bits/mathcalls.h" 3
extern _Float128 j0f128(_Float128) throw(); extern _Float128 __j0f128(_Float128) throw(); 
# 218
extern _Float128 j1f128(_Float128) throw(); extern _Float128 __j1f128(_Float128) throw(); 
# 219
extern _Float128 jnf128(int, _Float128) throw(); extern _Float128 __jnf128(int, _Float128) throw(); 
# 220
extern _Float128 y0f128(_Float128) throw(); extern _Float128 __y0f128(_Float128) throw(); 
# 221
extern _Float128 y1f128(_Float128) throw(); extern _Float128 __y1f128(_Float128) throw(); 
# 222
extern _Float128 ynf128(int, _Float128) throw(); extern _Float128 __ynf128(int, _Float128) throw(); 
# 228
extern _Float128 erff128(_Float128) throw(); extern _Float128 __erff128(_Float128) throw(); 
# 229
extern _Float128 erfcf128(_Float128) throw(); extern _Float128 __erfcf128(_Float128) throw(); 
# 230
extern _Float128 lgammaf128(_Float128) throw(); extern _Float128 __lgammaf128(_Float128) throw(); 
# 235
extern _Float128 tgammaf128(_Float128) throw(); extern _Float128 __tgammaf128(_Float128) throw(); 
# 249 "/usr/include/bits/mathcalls.h" 3
extern _Float128 lgammaf128_r(_Float128, int * __signgamp) throw(); extern _Float128 __lgammaf128_r(_Float128, int * __signgamp) throw(); 
# 256
extern _Float128 rintf128(_Float128 __x) throw(); extern _Float128 __rintf128(_Float128 __x) throw(); 
# 259
extern _Float128 nextafterf128(_Float128 __x, _Float128 __y) throw(); extern _Float128 __nextafterf128(_Float128 __x, _Float128 __y) throw(); 
# 266
extern _Float128 nextdownf128(_Float128 __x) throw(); extern _Float128 __nextdownf128(_Float128 __x) throw(); 
# 268
extern _Float128 nextupf128(_Float128 __x) throw(); extern _Float128 __nextupf128(_Float128 __x) throw(); 
# 272
extern _Float128 remainderf128(_Float128 __x, _Float128 __y) throw(); extern _Float128 __remainderf128(_Float128 __x, _Float128 __y) throw(); 
# 276
extern _Float128 scalbnf128(_Float128 __x, int __n) throw(); extern _Float128 __scalbnf128(_Float128 __x, int __n) throw(); 
# 280
extern int ilogbf128(_Float128 __x) throw(); extern int __ilogbf128(_Float128 __x) throw(); 
# 285
extern long llogbf128(_Float128 __x) throw(); extern long __llogbf128(_Float128 __x) throw(); 
# 290
extern _Float128 scalblnf128(_Float128 __x, long __n) throw(); extern _Float128 __scalblnf128(_Float128 __x, long __n) throw(); 
# 294
extern _Float128 nearbyintf128(_Float128 __x) throw(); extern _Float128 __nearbyintf128(_Float128 __x) throw(); 
# 298
extern _Float128 roundf128(_Float128 __x) throw() __attribute((const)); extern _Float128 __roundf128(_Float128 __x) throw() __attribute((const)); 
# 302
extern _Float128 truncf128(_Float128 __x) throw() __attribute((const)); extern _Float128 __truncf128(_Float128 __x) throw() __attribute((const)); 
# 307
extern _Float128 remquof128(_Float128 __x, _Float128 __y, int * __quo) throw(); extern _Float128 __remquof128(_Float128 __x, _Float128 __y, int * __quo) throw(); 
# 314
extern long lrintf128(_Float128 __x) throw(); extern long __lrintf128(_Float128 __x) throw(); 
# 316
__extension__ extern long long llrintf128(_Float128 __x) throw(); extern long long __llrintf128(_Float128 __x) throw(); 
# 320
extern long lroundf128(_Float128 __x) throw(); extern long __lroundf128(_Float128 __x) throw(); 
# 322
__extension__ extern long long llroundf128(_Float128 __x) throw(); extern long long __llroundf128(_Float128 __x) throw(); 
# 326
extern _Float128 fdimf128(_Float128 __x, _Float128 __y) throw(); extern _Float128 __fdimf128(_Float128 __x, _Float128 __y) throw(); 
# 329
extern _Float128 fmaxf128(_Float128 __x, _Float128 __y) throw() __attribute((const)); extern _Float128 __fmaxf128(_Float128 __x, _Float128 __y) throw() __attribute((const)); 
# 332
extern _Float128 fminf128(_Float128 __x, _Float128 __y) throw() __attribute((const)); extern _Float128 __fminf128(_Float128 __x, _Float128 __y) throw() __attribute((const)); 
# 335
extern _Float128 fmaf128(_Float128 __x, _Float128 __y, _Float128 __z) throw(); extern _Float128 __fmaf128(_Float128 __x, _Float128 __y, _Float128 __z) throw(); 
# 340
extern _Float128 roundevenf128(_Float128 __x) throw() __attribute((const)); extern _Float128 __roundevenf128(_Float128 __x) throw() __attribute((const)); 
# 344
extern __intmax_t fromfpf128(_Float128 __x, int __round, unsigned __width) throw(); extern __intmax_t __fromfpf128(_Float128 __x, int __round, unsigned __width) throw(); 
# 349
extern __uintmax_t ufromfpf128(_Float128 __x, int __round, unsigned __width) throw(); extern __uintmax_t __ufromfpf128(_Float128 __x, int __round, unsigned __width) throw(); 
# 355
extern __intmax_t fromfpxf128(_Float128 __x, int __round, unsigned __width) throw(); extern __intmax_t __fromfpxf128(_Float128 __x, int __round, unsigned __width) throw(); 
# 361
extern __uintmax_t ufromfpxf128(_Float128 __x, int __round, unsigned __width) throw(); extern __uintmax_t __ufromfpxf128(_Float128 __x, int __round, unsigned __width) throw(); 
# 365
extern _Float128 fmaxmagf128(_Float128 __x, _Float128 __y) throw() __attribute((const)); extern _Float128 __fmaxmagf128(_Float128 __x, _Float128 __y) throw() __attribute((const)); 
# 368
extern _Float128 fminmagf128(_Float128 __x, _Float128 __y) throw() __attribute((const)); extern _Float128 __fminmagf128(_Float128 __x, _Float128 __y) throw() __attribute((const)); 
# 371
extern int totalorderf128(_Float128 __x, _Float128 __y) throw()
# 372
 __attribute((const)); 
# 375
extern int totalordermagf128(_Float128 __x, _Float128 __y) throw()
# 376
 __attribute((const)); 
# 379
extern int canonicalizef128(_Float128 * __cx, const _Float128 * __x) throw(); 
# 382
extern _Float128 getpayloadf128(const _Float128 * __x) throw(); extern _Float128 __getpayloadf128(const _Float128 * __x) throw(); 
# 385
extern int setpayloadf128(_Float128 * __x, _Float128 __payload) throw(); 
# 388
extern int setpayloadsigf128(_Float128 * __x, _Float128 __payload) throw(); 
# 53 "/usr/include/bits/mathcalls.h" 3
extern _Float32x acosf32x(_Float32x __x) throw(); extern _Float32x __acosf32x(_Float32x __x) throw(); 
# 55
extern _Float32x asinf32x(_Float32x __x) throw(); extern _Float32x __asinf32x(_Float32x __x) throw(); 
# 57
extern _Float32x atanf32x(_Float32x __x) throw(); extern _Float32x __atanf32x(_Float32x __x) throw(); 
# 59
extern _Float32x atan2f32x(_Float32x __y, _Float32x __x) throw(); extern _Float32x __atan2f32x(_Float32x __y, _Float32x __x) throw(); 
# 62
extern _Float32x cosf32x(_Float32x __x) throw(); extern _Float32x __cosf32x(_Float32x __x) throw(); 
# 64
extern _Float32x sinf32x(_Float32x __x) throw(); extern _Float32x __sinf32x(_Float32x __x) throw(); 
# 66
extern _Float32x tanf32x(_Float32x __x) throw(); extern _Float32x __tanf32x(_Float32x __x) throw(); 
# 71
extern _Float32x coshf32x(_Float32x __x) throw(); extern _Float32x __coshf32x(_Float32x __x) throw(); 
# 73
extern _Float32x sinhf32x(_Float32x __x) throw(); extern _Float32x __sinhf32x(_Float32x __x) throw(); 
# 75
extern _Float32x tanhf32x(_Float32x __x) throw(); extern _Float32x __tanhf32x(_Float32x __x) throw(); 
# 79
extern void sincosf32x(_Float32x __x, _Float32x * __sinx, _Float32x * __cosx) throw(); extern void __sincosf32x(_Float32x __x, _Float32x * __sinx, _Float32x * __cosx) throw(); 
# 85
extern _Float32x acoshf32x(_Float32x __x) throw(); extern _Float32x __acoshf32x(_Float32x __x) throw(); 
# 87
extern _Float32x asinhf32x(_Float32x __x) throw(); extern _Float32x __asinhf32x(_Float32x __x) throw(); 
# 89
extern _Float32x atanhf32x(_Float32x __x) throw(); extern _Float32x __atanhf32x(_Float32x __x) throw(); 
# 95
extern _Float32x expf32x(_Float32x __x) throw(); extern _Float32x __expf32x(_Float32x __x) throw(); 
# 98
extern _Float32x frexpf32x(_Float32x __x, int * __exponent) throw(); extern _Float32x __frexpf32x(_Float32x __x, int * __exponent) throw(); 
# 101
extern _Float32x ldexpf32x(_Float32x __x, int __exponent) throw(); extern _Float32x __ldexpf32x(_Float32x __x, int __exponent) throw(); 
# 104
extern _Float32x logf32x(_Float32x __x) throw(); extern _Float32x __logf32x(_Float32x __x) throw(); 
# 107
extern _Float32x log10f32x(_Float32x __x) throw(); extern _Float32x __log10f32x(_Float32x __x) throw(); 
# 110
extern _Float32x modff32x(_Float32x __x, _Float32x * __iptr) throw(); extern _Float32x __modff32x(_Float32x __x, _Float32x * __iptr) throw() __attribute((__nonnull__(2))); 
# 114
extern _Float32x exp10f32x(_Float32x __x) throw(); extern _Float32x __exp10f32x(_Float32x __x) throw(); 
# 119
extern _Float32x expm1f32x(_Float32x __x) throw(); extern _Float32x __expm1f32x(_Float32x __x) throw(); 
# 122
extern _Float32x log1pf32x(_Float32x __x) throw(); extern _Float32x __log1pf32x(_Float32x __x) throw(); 
# 125
extern _Float32x logbf32x(_Float32x __x) throw(); extern _Float32x __logbf32x(_Float32x __x) throw(); 
# 130
extern _Float32x exp2f32x(_Float32x __x) throw(); extern _Float32x __exp2f32x(_Float32x __x) throw(); 
# 133
extern _Float32x log2f32x(_Float32x __x) throw(); extern _Float32x __log2f32x(_Float32x __x) throw(); 
# 140
extern _Float32x powf32x(_Float32x __x, _Float32x __y) throw(); extern _Float32x __powf32x(_Float32x __x, _Float32x __y) throw(); 
# 143
extern _Float32x sqrtf32x(_Float32x __x) throw(); extern _Float32x __sqrtf32x(_Float32x __x) throw(); 
# 147
extern _Float32x hypotf32x(_Float32x __x, _Float32x __y) throw(); extern _Float32x __hypotf32x(_Float32x __x, _Float32x __y) throw(); 
# 152
extern _Float32x cbrtf32x(_Float32x __x) throw(); extern _Float32x __cbrtf32x(_Float32x __x) throw(); 
# 159
extern _Float32x ceilf32x(_Float32x __x) throw() __attribute((const)); extern _Float32x __ceilf32x(_Float32x __x) throw() __attribute((const)); 
# 162
extern _Float32x fabsf32x(_Float32x __x) throw() __attribute((const)); extern _Float32x __fabsf32x(_Float32x __x) throw() __attribute((const)); 
# 165
extern _Float32x floorf32x(_Float32x __x) throw() __attribute((const)); extern _Float32x __floorf32x(_Float32x __x) throw() __attribute((const)); 
# 168
extern _Float32x fmodf32x(_Float32x __x, _Float32x __y) throw(); extern _Float32x __fmodf32x(_Float32x __x, _Float32x __y) throw(); 
# 196 "/usr/include/bits/mathcalls.h" 3
extern _Float32x copysignf32x(_Float32x __x, _Float32x __y) throw() __attribute((const)); extern _Float32x __copysignf32x(_Float32x __x, _Float32x __y) throw() __attribute((const)); 
# 201
extern _Float32x nanf32x(const char * __tagb) throw(); extern _Float32x __nanf32x(const char * __tagb) throw(); 
# 217 "/usr/include/bits/mathcalls.h" 3
extern _Float32x j0f32x(_Float32x) throw(); extern _Float32x __j0f32x(_Float32x) throw(); 
# 218
extern _Float32x j1f32x(_Float32x) throw(); extern _Float32x __j1f32x(_Float32x) throw(); 
# 219
extern _Float32x jnf32x(int, _Float32x) throw(); extern _Float32x __jnf32x(int, _Float32x) throw(); 
# 220
extern _Float32x y0f32x(_Float32x) throw(); extern _Float32x __y0f32x(_Float32x) throw(); 
# 221
extern _Float32x y1f32x(_Float32x) throw(); extern _Float32x __y1f32x(_Float32x) throw(); 
# 222
extern _Float32x ynf32x(int, _Float32x) throw(); extern _Float32x __ynf32x(int, _Float32x) throw(); 
# 228
extern _Float32x erff32x(_Float32x) throw(); extern _Float32x __erff32x(_Float32x) throw(); 
# 229
extern _Float32x erfcf32x(_Float32x) throw(); extern _Float32x __erfcf32x(_Float32x) throw(); 
# 230
extern _Float32x lgammaf32x(_Float32x) throw(); extern _Float32x __lgammaf32x(_Float32x) throw(); 
# 235
extern _Float32x tgammaf32x(_Float32x) throw(); extern _Float32x __tgammaf32x(_Float32x) throw(); 
# 249 "/usr/include/bits/mathcalls.h" 3
extern _Float32x lgammaf32x_r(_Float32x, int * __signgamp) throw(); extern _Float32x __lgammaf32x_r(_Float32x, int * __signgamp) throw(); 
# 256
extern _Float32x rintf32x(_Float32x __x) throw(); extern _Float32x __rintf32x(_Float32x __x) throw(); 
# 259
extern _Float32x nextafterf32x(_Float32x __x, _Float32x __y) throw(); extern _Float32x __nextafterf32x(_Float32x __x, _Float32x __y) throw(); 
# 266
extern _Float32x nextdownf32x(_Float32x __x) throw(); extern _Float32x __nextdownf32x(_Float32x __x) throw(); 
# 268
extern _Float32x nextupf32x(_Float32x __x) throw(); extern _Float32x __nextupf32x(_Float32x __x) throw(); 
# 272
extern _Float32x remainderf32x(_Float32x __x, _Float32x __y) throw(); extern _Float32x __remainderf32x(_Float32x __x, _Float32x __y) throw(); 
# 276
extern _Float32x scalbnf32x(_Float32x __x, int __n) throw(); extern _Float32x __scalbnf32x(_Float32x __x, int __n) throw(); 
# 280
extern int ilogbf32x(_Float32x __x) throw(); extern int __ilogbf32x(_Float32x __x) throw(); 
# 285
extern long llogbf32x(_Float32x __x) throw(); extern long __llogbf32x(_Float32x __x) throw(); 
# 290
extern _Float32x scalblnf32x(_Float32x __x, long __n) throw(); extern _Float32x __scalblnf32x(_Float32x __x, long __n) throw(); 
# 294
extern _Float32x nearbyintf32x(_Float32x __x) throw(); extern _Float32x __nearbyintf32x(_Float32x __x) throw(); 
# 298
extern _Float32x roundf32x(_Float32x __x) throw() __attribute((const)); extern _Float32x __roundf32x(_Float32x __x) throw() __attribute((const)); 
# 302
extern _Float32x truncf32x(_Float32x __x) throw() __attribute((const)); extern _Float32x __truncf32x(_Float32x __x) throw() __attribute((const)); 
# 307
extern _Float32x remquof32x(_Float32x __x, _Float32x __y, int * __quo) throw(); extern _Float32x __remquof32x(_Float32x __x, _Float32x __y, int * __quo) throw(); 
# 314
extern long lrintf32x(_Float32x __x) throw(); extern long __lrintf32x(_Float32x __x) throw(); 
# 316
__extension__ extern long long llrintf32x(_Float32x __x) throw(); extern long long __llrintf32x(_Float32x __x) throw(); 
# 320
extern long lroundf32x(_Float32x __x) throw(); extern long __lroundf32x(_Float32x __x) throw(); 
# 322
__extension__ extern long long llroundf32x(_Float32x __x) throw(); extern long long __llroundf32x(_Float32x __x) throw(); 
# 326
extern _Float32x fdimf32x(_Float32x __x, _Float32x __y) throw(); extern _Float32x __fdimf32x(_Float32x __x, _Float32x __y) throw(); 
# 329
extern _Float32x fmaxf32x(_Float32x __x, _Float32x __y) throw() __attribute((const)); extern _Float32x __fmaxf32x(_Float32x __x, _Float32x __y) throw() __attribute((const)); 
# 332
extern _Float32x fminf32x(_Float32x __x, _Float32x __y) throw() __attribute((const)); extern _Float32x __fminf32x(_Float32x __x, _Float32x __y) throw() __attribute((const)); 
# 335
extern _Float32x fmaf32x(_Float32x __x, _Float32x __y, _Float32x __z) throw(); extern _Float32x __fmaf32x(_Float32x __x, _Float32x __y, _Float32x __z) throw(); 
# 340
extern _Float32x roundevenf32x(_Float32x __x) throw() __attribute((const)); extern _Float32x __roundevenf32x(_Float32x __x) throw() __attribute((const)); 
# 344
extern __intmax_t fromfpf32x(_Float32x __x, int __round, unsigned __width) throw(); extern __intmax_t __fromfpf32x(_Float32x __x, int __round, unsigned __width) throw(); 
# 349
extern __uintmax_t ufromfpf32x(_Float32x __x, int __round, unsigned __width) throw(); extern __uintmax_t __ufromfpf32x(_Float32x __x, int __round, unsigned __width) throw(); 
# 355
extern __intmax_t fromfpxf32x(_Float32x __x, int __round, unsigned __width) throw(); extern __intmax_t __fromfpxf32x(_Float32x __x, int __round, unsigned __width) throw(); 
# 361
extern __uintmax_t ufromfpxf32x(_Float32x __x, int __round, unsigned __width) throw(); extern __uintmax_t __ufromfpxf32x(_Float32x __x, int __round, unsigned __width) throw(); 
# 365
extern _Float32x fmaxmagf32x(_Float32x __x, _Float32x __y) throw() __attribute((const)); extern _Float32x __fmaxmagf32x(_Float32x __x, _Float32x __y) throw() __attribute((const)); 
# 368
extern _Float32x fminmagf32x(_Float32x __x, _Float32x __y) throw() __attribute((const)); extern _Float32x __fminmagf32x(_Float32x __x, _Float32x __y) throw() __attribute((const)); 
# 371
extern int totalorderf32x(_Float32x __x, _Float32x __y) throw()
# 372
 __attribute((const)); 
# 375
extern int totalordermagf32x(_Float32x __x, _Float32x __y) throw()
# 376
 __attribute((const)); 
# 379
extern int canonicalizef32x(_Float32x * __cx, const _Float32x * __x) throw(); 
# 382
extern _Float32x getpayloadf32x(const _Float32x * __x) throw(); extern _Float32x __getpayloadf32x(const _Float32x * __x) throw(); 
# 385
extern int setpayloadf32x(_Float32x * __x, _Float32x __payload) throw(); 
# 388
extern int setpayloadsigf32x(_Float32x * __x, _Float32x __payload) throw(); 
# 53 "/usr/include/bits/mathcalls.h" 3
extern _Float64x acosf64x(_Float64x __x) throw(); extern _Float64x __acosf64x(_Float64x __x) throw(); 
# 55
extern _Float64x asinf64x(_Float64x __x) throw(); extern _Float64x __asinf64x(_Float64x __x) throw(); 
# 57
extern _Float64x atanf64x(_Float64x __x) throw(); extern _Float64x __atanf64x(_Float64x __x) throw(); 
# 59
extern _Float64x atan2f64x(_Float64x __y, _Float64x __x) throw(); extern _Float64x __atan2f64x(_Float64x __y, _Float64x __x) throw(); 
# 62
extern _Float64x cosf64x(_Float64x __x) throw(); extern _Float64x __cosf64x(_Float64x __x) throw(); 
# 64
extern _Float64x sinf64x(_Float64x __x) throw(); extern _Float64x __sinf64x(_Float64x __x) throw(); 
# 66
extern _Float64x tanf64x(_Float64x __x) throw(); extern _Float64x __tanf64x(_Float64x __x) throw(); 
# 71
extern _Float64x coshf64x(_Float64x __x) throw(); extern _Float64x __coshf64x(_Float64x __x) throw(); 
# 73
extern _Float64x sinhf64x(_Float64x __x) throw(); extern _Float64x __sinhf64x(_Float64x __x) throw(); 
# 75
extern _Float64x tanhf64x(_Float64x __x) throw(); extern _Float64x __tanhf64x(_Float64x __x) throw(); 
# 79
extern void sincosf64x(_Float64x __x, _Float64x * __sinx, _Float64x * __cosx) throw(); extern void __sincosf64x(_Float64x __x, _Float64x * __sinx, _Float64x * __cosx) throw(); 
# 85
extern _Float64x acoshf64x(_Float64x __x) throw(); extern _Float64x __acoshf64x(_Float64x __x) throw(); 
# 87
extern _Float64x asinhf64x(_Float64x __x) throw(); extern _Float64x __asinhf64x(_Float64x __x) throw(); 
# 89
extern _Float64x atanhf64x(_Float64x __x) throw(); extern _Float64x __atanhf64x(_Float64x __x) throw(); 
# 95
extern _Float64x expf64x(_Float64x __x) throw(); extern _Float64x __expf64x(_Float64x __x) throw(); 
# 98
extern _Float64x frexpf64x(_Float64x __x, int * __exponent) throw(); extern _Float64x __frexpf64x(_Float64x __x, int * __exponent) throw(); 
# 101
extern _Float64x ldexpf64x(_Float64x __x, int __exponent) throw(); extern _Float64x __ldexpf64x(_Float64x __x, int __exponent) throw(); 
# 104
extern _Float64x logf64x(_Float64x __x) throw(); extern _Float64x __logf64x(_Float64x __x) throw(); 
# 107
extern _Float64x log10f64x(_Float64x __x) throw(); extern _Float64x __log10f64x(_Float64x __x) throw(); 
# 110
extern _Float64x modff64x(_Float64x __x, _Float64x * __iptr) throw(); extern _Float64x __modff64x(_Float64x __x, _Float64x * __iptr) throw() __attribute((__nonnull__(2))); 
# 114
extern _Float64x exp10f64x(_Float64x __x) throw(); extern _Float64x __exp10f64x(_Float64x __x) throw(); 
# 119
extern _Float64x expm1f64x(_Float64x __x) throw(); extern _Float64x __expm1f64x(_Float64x __x) throw(); 
# 122
extern _Float64x log1pf64x(_Float64x __x) throw(); extern _Float64x __log1pf64x(_Float64x __x) throw(); 
# 125
extern _Float64x logbf64x(_Float64x __x) throw(); extern _Float64x __logbf64x(_Float64x __x) throw(); 
# 130
extern _Float64x exp2f64x(_Float64x __x) throw(); extern _Float64x __exp2f64x(_Float64x __x) throw(); 
# 133
extern _Float64x log2f64x(_Float64x __x) throw(); extern _Float64x __log2f64x(_Float64x __x) throw(); 
# 140
extern _Float64x powf64x(_Float64x __x, _Float64x __y) throw(); extern _Float64x __powf64x(_Float64x __x, _Float64x __y) throw(); 
# 143
extern _Float64x sqrtf64x(_Float64x __x) throw(); extern _Float64x __sqrtf64x(_Float64x __x) throw(); 
# 147
extern _Float64x hypotf64x(_Float64x __x, _Float64x __y) throw(); extern _Float64x __hypotf64x(_Float64x __x, _Float64x __y) throw(); 
# 152
extern _Float64x cbrtf64x(_Float64x __x) throw(); extern _Float64x __cbrtf64x(_Float64x __x) throw(); 
# 159
extern _Float64x ceilf64x(_Float64x __x) throw() __attribute((const)); extern _Float64x __ceilf64x(_Float64x __x) throw() __attribute((const)); 
# 162
extern _Float64x fabsf64x(_Float64x __x) throw() __attribute((const)); extern _Float64x __fabsf64x(_Float64x __x) throw() __attribute((const)); 
# 165
extern _Float64x floorf64x(_Float64x __x) throw() __attribute((const)); extern _Float64x __floorf64x(_Float64x __x) throw() __attribute((const)); 
# 168
extern _Float64x fmodf64x(_Float64x __x, _Float64x __y) throw(); extern _Float64x __fmodf64x(_Float64x __x, _Float64x __y) throw(); 
# 196 "/usr/include/bits/mathcalls.h" 3
extern _Float64x copysignf64x(_Float64x __x, _Float64x __y) throw() __attribute((const)); extern _Float64x __copysignf64x(_Float64x __x, _Float64x __y) throw() __attribute((const)); 
# 201
extern _Float64x nanf64x(const char * __tagb) throw(); extern _Float64x __nanf64x(const char * __tagb) throw(); 
# 217 "/usr/include/bits/mathcalls.h" 3
extern _Float64x j0f64x(_Float64x) throw(); extern _Float64x __j0f64x(_Float64x) throw(); 
# 218
extern _Float64x j1f64x(_Float64x) throw(); extern _Float64x __j1f64x(_Float64x) throw(); 
# 219
extern _Float64x jnf64x(int, _Float64x) throw(); extern _Float64x __jnf64x(int, _Float64x) throw(); 
# 220
extern _Float64x y0f64x(_Float64x) throw(); extern _Float64x __y0f64x(_Float64x) throw(); 
# 221
extern _Float64x y1f64x(_Float64x) throw(); extern _Float64x __y1f64x(_Float64x) throw(); 
# 222
extern _Float64x ynf64x(int, _Float64x) throw(); extern _Float64x __ynf64x(int, _Float64x) throw(); 
# 228
extern _Float64x erff64x(_Float64x) throw(); extern _Float64x __erff64x(_Float64x) throw(); 
# 229
extern _Float64x erfcf64x(_Float64x) throw(); extern _Float64x __erfcf64x(_Float64x) throw(); 
# 230
extern _Float64x lgammaf64x(_Float64x) throw(); extern _Float64x __lgammaf64x(_Float64x) throw(); 
# 235
extern _Float64x tgammaf64x(_Float64x) throw(); extern _Float64x __tgammaf64x(_Float64x) throw(); 
# 249 "/usr/include/bits/mathcalls.h" 3
extern _Float64x lgammaf64x_r(_Float64x, int * __signgamp) throw(); extern _Float64x __lgammaf64x_r(_Float64x, int * __signgamp) throw(); 
# 256
extern _Float64x rintf64x(_Float64x __x) throw(); extern _Float64x __rintf64x(_Float64x __x) throw(); 
# 259
extern _Float64x nextafterf64x(_Float64x __x, _Float64x __y) throw(); extern _Float64x __nextafterf64x(_Float64x __x, _Float64x __y) throw(); 
# 266
extern _Float64x nextdownf64x(_Float64x __x) throw(); extern _Float64x __nextdownf64x(_Float64x __x) throw(); 
# 268
extern _Float64x nextupf64x(_Float64x __x) throw(); extern _Float64x __nextupf64x(_Float64x __x) throw(); 
# 272
extern _Float64x remainderf64x(_Float64x __x, _Float64x __y) throw(); extern _Float64x __remainderf64x(_Float64x __x, _Float64x __y) throw(); 
# 276
extern _Float64x scalbnf64x(_Float64x __x, int __n) throw(); extern _Float64x __scalbnf64x(_Float64x __x, int __n) throw(); 
# 280
extern int ilogbf64x(_Float64x __x) throw(); extern int __ilogbf64x(_Float64x __x) throw(); 
# 285
extern long llogbf64x(_Float64x __x) throw(); extern long __llogbf64x(_Float64x __x) throw(); 
# 290
extern _Float64x scalblnf64x(_Float64x __x, long __n) throw(); extern _Float64x __scalblnf64x(_Float64x __x, long __n) throw(); 
# 294
extern _Float64x nearbyintf64x(_Float64x __x) throw(); extern _Float64x __nearbyintf64x(_Float64x __x) throw(); 
# 298
extern _Float64x roundf64x(_Float64x __x) throw() __attribute((const)); extern _Float64x __roundf64x(_Float64x __x) throw() __attribute((const)); 
# 302
extern _Float64x truncf64x(_Float64x __x) throw() __attribute((const)); extern _Float64x __truncf64x(_Float64x __x) throw() __attribute((const)); 
# 307
extern _Float64x remquof64x(_Float64x __x, _Float64x __y, int * __quo) throw(); extern _Float64x __remquof64x(_Float64x __x, _Float64x __y, int * __quo) throw(); 
# 314
extern long lrintf64x(_Float64x __x) throw(); extern long __lrintf64x(_Float64x __x) throw(); 
# 316
__extension__ extern long long llrintf64x(_Float64x __x) throw(); extern long long __llrintf64x(_Float64x __x) throw(); 
# 320
extern long lroundf64x(_Float64x __x) throw(); extern long __lroundf64x(_Float64x __x) throw(); 
# 322
__extension__ extern long long llroundf64x(_Float64x __x) throw(); extern long long __llroundf64x(_Float64x __x) throw(); 
# 326
extern _Float64x fdimf64x(_Float64x __x, _Float64x __y) throw(); extern _Float64x __fdimf64x(_Float64x __x, _Float64x __y) throw(); 
# 329
extern _Float64x fmaxf64x(_Float64x __x, _Float64x __y) throw() __attribute((const)); extern _Float64x __fmaxf64x(_Float64x __x, _Float64x __y) throw() __attribute((const)); 
# 332
extern _Float64x fminf64x(_Float64x __x, _Float64x __y) throw() __attribute((const)); extern _Float64x __fminf64x(_Float64x __x, _Float64x __y) throw() __attribute((const)); 
# 335
extern _Float64x fmaf64x(_Float64x __x, _Float64x __y, _Float64x __z) throw(); extern _Float64x __fmaf64x(_Float64x __x, _Float64x __y, _Float64x __z) throw(); 
# 340
extern _Float64x roundevenf64x(_Float64x __x) throw() __attribute((const)); extern _Float64x __roundevenf64x(_Float64x __x) throw() __attribute((const)); 
# 344
extern __intmax_t fromfpf64x(_Float64x __x, int __round, unsigned __width) throw(); extern __intmax_t __fromfpf64x(_Float64x __x, int __round, unsigned __width) throw(); 
# 349
extern __uintmax_t ufromfpf64x(_Float64x __x, int __round, unsigned __width) throw(); extern __uintmax_t __ufromfpf64x(_Float64x __x, int __round, unsigned __width) throw(); 
# 355
extern __intmax_t fromfpxf64x(_Float64x __x, int __round, unsigned __width) throw(); extern __intmax_t __fromfpxf64x(_Float64x __x, int __round, unsigned __width) throw(); 
# 361
extern __uintmax_t ufromfpxf64x(_Float64x __x, int __round, unsigned __width) throw(); extern __uintmax_t __ufromfpxf64x(_Float64x __x, int __round, unsigned __width) throw(); 
# 365
extern _Float64x fmaxmagf64x(_Float64x __x, _Float64x __y) throw() __attribute((const)); extern _Float64x __fmaxmagf64x(_Float64x __x, _Float64x __y) throw() __attribute((const)); 
# 368
extern _Float64x fminmagf64x(_Float64x __x, _Float64x __y) throw() __attribute((const)); extern _Float64x __fminmagf64x(_Float64x __x, _Float64x __y) throw() __attribute((const)); 
# 371
extern int totalorderf64x(_Float64x __x, _Float64x __y) throw()
# 372
 __attribute((const)); 
# 375
extern int totalordermagf64x(_Float64x __x, _Float64x __y) throw()
# 376
 __attribute((const)); 
# 379
extern int canonicalizef64x(_Float64x * __cx, const _Float64x * __x) throw(); 
# 382
extern _Float64x getpayloadf64x(const _Float64x * __x) throw(); extern _Float64x __getpayloadf64x(const _Float64x * __x) throw(); 
# 385
extern int setpayloadf64x(_Float64x * __x, _Float64x __payload) throw(); 
# 388
extern int setpayloadsigf64x(_Float64x * __x, _Float64x __payload) throw(); 
# 24 "/usr/include/bits/mathcalls-narrow.h" 3
extern float fadd(double __x, double __y) throw(); 
# 27
extern float fdiv(double __x, double __y) throw(); 
# 30
extern float fmul(double __x, double __y) throw(); 
# 33
extern float fsub(double __x, double __y) throw(); 
# 24 "/usr/include/bits/mathcalls-narrow.h" 3
extern float faddl(long double __x, long double __y) throw(); 
# 27
extern float fdivl(long double __x, long double __y) throw(); 
# 30
extern float fmull(long double __x, long double __y) throw(); 
# 33
extern float fsubl(long double __x, long double __y) throw(); 
# 24 "/usr/include/bits/mathcalls-narrow.h" 3
extern double daddl(long double __x, long double __y) throw(); 
# 27
extern double ddivl(long double __x, long double __y) throw(); 
# 30
extern double dmull(long double __x, long double __y) throw(); 
# 33
extern double dsubl(long double __x, long double __y) throw(); 
# 24 "/usr/include/bits/mathcalls-narrow.h" 3
extern _Float32 f32addf32x(_Float32x __x, _Float32x __y) throw(); 
# 27
extern _Float32 f32divf32x(_Float32x __x, _Float32x __y) throw(); 
# 30
extern _Float32 f32mulf32x(_Float32x __x, _Float32x __y) throw(); 
# 33
extern _Float32 f32subf32x(_Float32x __x, _Float32x __y) throw(); 
# 24 "/usr/include/bits/mathcalls-narrow.h" 3
extern _Float32 f32addf64(_Float64 __x, _Float64 __y) throw(); 
# 27
extern _Float32 f32divf64(_Float64 __x, _Float64 __y) throw(); 
# 30
extern _Float32 f32mulf64(_Float64 __x, _Float64 __y) throw(); 
# 33
extern _Float32 f32subf64(_Float64 __x, _Float64 __y) throw(); 
# 24 "/usr/include/bits/mathcalls-narrow.h" 3
extern _Float32 f32addf64x(_Float64x __x, _Float64x __y) throw(); 
# 27
extern _Float32 f32divf64x(_Float64x __x, _Float64x __y) throw(); 
# 30
extern _Float32 f32mulf64x(_Float64x __x, _Float64x __y) throw(); 
# 33
extern _Float32 f32subf64x(_Float64x __x, _Float64x __y) throw(); 
# 24 "/usr/include/bits/mathcalls-narrow.h" 3
extern _Float32 f32addf128(_Float128 __x, _Float128 __y) throw(); 
# 27
extern _Float32 f32divf128(_Float128 __x, _Float128 __y) throw(); 
# 30
extern _Float32 f32mulf128(_Float128 __x, _Float128 __y) throw(); 
# 33
extern _Float32 f32subf128(_Float128 __x, _Float128 __y) throw(); 
# 24 "/usr/include/bits/mathcalls-narrow.h" 3
extern _Float32x f32xaddf64(_Float64 __x, _Float64 __y) throw(); 
# 27
extern _Float32x f32xdivf64(_Float64 __x, _Float64 __y) throw(); 
# 30
extern _Float32x f32xmulf64(_Float64 __x, _Float64 __y) throw(); 
# 33
extern _Float32x f32xsubf64(_Float64 __x, _Float64 __y) throw(); 
# 24 "/usr/include/bits/mathcalls-narrow.h" 3
extern _Float32x f32xaddf64x(_Float64x __x, _Float64x __y) throw(); 
# 27
extern _Float32x f32xdivf64x(_Float64x __x, _Float64x __y) throw(); 
# 30
extern _Float32x f32xmulf64x(_Float64x __x, _Float64x __y) throw(); 
# 33
extern _Float32x f32xsubf64x(_Float64x __x, _Float64x __y) throw(); 
# 24 "/usr/include/bits/mathcalls-narrow.h" 3
extern _Float32x f32xaddf128(_Float128 __x, _Float128 __y) throw(); 
# 27
extern _Float32x f32xdivf128(_Float128 __x, _Float128 __y) throw(); 
# 30
extern _Float32x f32xmulf128(_Float128 __x, _Float128 __y) throw(); 
# 33
extern _Float32x f32xsubf128(_Float128 __x, _Float128 __y) throw(); 
# 24 "/usr/include/bits/mathcalls-narrow.h" 3
extern _Float64 f64addf64x(_Float64x __x, _Float64x __y) throw(); 
# 27
extern _Float64 f64divf64x(_Float64x __x, _Float64x __y) throw(); 
# 30
extern _Float64 f64mulf64x(_Float64x __x, _Float64x __y) throw(); 
# 33
extern _Float64 f64subf64x(_Float64x __x, _Float64x __y) throw(); 
# 24 "/usr/include/bits/mathcalls-narrow.h" 3
extern _Float64 f64addf128(_Float128 __x, _Float128 __y) throw(); 
# 27
extern _Float64 f64divf128(_Float128 __x, _Float128 __y) throw(); 
# 30
extern _Float64 f64mulf128(_Float128 __x, _Float128 __y) throw(); 
# 33
extern _Float64 f64subf128(_Float128 __x, _Float128 __y) throw(); 
# 24 "/usr/include/bits/mathcalls-narrow.h" 3
extern _Float64x f64xaddf128(_Float128 __x, _Float128 __y) throw(); 
# 27
extern _Float64x f64xdivf128(_Float128 __x, _Float128 __y) throw(); 
# 30
extern _Float64x f64xmulf128(_Float128 __x, _Float128 __y) throw(); 
# 33
extern _Float64x f64xsubf128(_Float128 __x, _Float128 __y) throw(); 
# 773 "/usr/include/math.h" 3
extern int signgam; 
# 854 "/usr/include/math.h" 3
enum { 
# 855
FP_NAN, 
# 858
FP_INFINITE, 
# 861
FP_ZERO, 
# 864
FP_SUBNORMAL, 
# 867
FP_NORMAL
# 870
}; 
# 23 "/usr/include/bits/iscanonical.h" 3
extern int __iscanonicall(long double __x) throw()
# 24
 __attribute((const)); 
# 46 "/usr/include/bits/iscanonical.h" 3
extern "C++" {
# 47
inline int iscanonical(float __val) { return (((void)((__typeof__(__val))__val)), 1); } 
# 48
inline int iscanonical(double __val) { return (((void)((__typeof__(__val))__val)), 1); } 
# 49
inline int iscanonical(long double __val) { return __iscanonicall(__val); } 
# 51
inline int iscanonical(_Float128 __val) { return (((void)((__typeof__(__val))__val)), 1); } 
# 53
}
# 985 "/usr/include/math.h" 3
extern "C++" {
# 986
inline int issignaling(float __val) { return __issignalingf(__val); } 
# 987
inline int issignaling(double __val) { return __issignaling(__val); } 
# 989
inline int issignaling(long double __val) 
# 990
{ 
# 994
return __issignalingl(__val); 
# 996
} 
# 1000
inline int issignaling(_Float128 __val) { return __issignalingf128(__val); } 
# 1002
}
# 1016 "/usr/include/math.h" 3
extern "C++" {
# 1047 "/usr/include/math.h" 3
template< class __T> inline bool 
# 1048
iszero(__T __val) 
# 1049
{ 
# 1050
return __val == 0; 
# 1051
} 
# 1053
}
# 1498 "/usr/include/math.h" 3
extern "C++" {
# 1499
template< class > struct __iseqsig_type; 
# 1501
template<> struct __iseqsig_type< float>  { 
# 1503
static int __call(float __x, float __y) throw() 
# 1504
{ 
# 1505
return __iseqsigf(__x, __y); 
# 1506
} 
# 1507
}; 
# 1509
template<> struct __iseqsig_type< double>  { 
# 1511
static int __call(double __x, double __y) throw() 
# 1512
{ 
# 1513
return __iseqsig(__x, __y); 
# 1514
} 
# 1515
}; 
# 1517
template<> struct __iseqsig_type< long double>  { 
# 1519
static int __call(long double __x, long double __y) throw() 
# 1520
{ 
# 1522
return __iseqsigl(__x, __y); 
# 1526
} 
# 1527
}; 
# 1532
template<> struct __iseqsig_type< __float128>  { 
# 1534
static int __call(_Float128 __x, _Float128 __y) throw() 
# 1535
{ 
# 1536
return __iseqsigf128(__x, __y); 
# 1537
} 
# 1538
}; 
# 1541
template< class _T1, class _T2> inline int 
# 1543
iseqsig(_T1 __x, _T2 __y) throw() 
# 1544
{ 
# 1546
typedef __decltype(((__x + __y) + (0.0F))) _T3; 
# 1550
return __iseqsig_type< __decltype(((__x + __y) + (0.0F)))> ::__call(__x, __y); 
# 1551
} 
# 1553
}
# 1558
}
# 33 "/usr/include/stdlib.h" 3
extern "C" {
# 62 "/usr/include/stdlib.h" 3
typedef 
# 59
struct { 
# 60
int quot; 
# 61
int rem; 
# 62
} div_t; 
# 70
typedef 
# 67
struct { 
# 68
long quot; 
# 69
long rem; 
# 70
} ldiv_t; 
# 80
__extension__ typedef 
# 77
struct { 
# 78
long long quot; 
# 79
long long rem; 
# 80
} lldiv_t; 
# 97 "/usr/include/stdlib.h" 3
extern size_t __ctype_get_mb_cur_max() throw(); 
# 101
extern double atof(const char * __nptr) throw()
# 102
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 104
extern int atoi(const char * __nptr) throw()
# 105
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 107
extern long atol(const char * __nptr) throw()
# 108
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 112
__extension__ extern long long atoll(const char * __nptr) throw()
# 113
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 117
extern double strtod(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 119
 __attribute((__nonnull__(1))); 
# 123
extern float strtof(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 124
 __attribute((__nonnull__(1))); 
# 126
extern long double strtold(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 128
 __attribute((__nonnull__(1))); 
# 140 "/usr/include/stdlib.h" 3
extern _Float32 strtof32(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 142
 __attribute((__nonnull__(1))); 
# 146
extern _Float64 strtof64(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 148
 __attribute((__nonnull__(1))); 
# 152
extern _Float128 strtof128(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 154
 __attribute((__nonnull__(1))); 
# 158
extern _Float32x strtof32x(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 160
 __attribute((__nonnull__(1))); 
# 164
extern _Float64x strtof64x(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 166
 __attribute((__nonnull__(1))); 
# 176 "/usr/include/stdlib.h" 3
extern long strtol(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 178
 __attribute((__nonnull__(1))); 
# 180
extern unsigned long strtoul(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 182
 __attribute((__nonnull__(1))); 
# 187
__extension__ extern long long strtoq(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 189
 __attribute((__nonnull__(1))); 
# 192
__extension__ extern unsigned long long strtouq(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 194
 __attribute((__nonnull__(1))); 
# 200
__extension__ extern long long strtoll(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 202
 __attribute((__nonnull__(1))); 
# 205
__extension__ extern unsigned long long strtoull(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 207
 __attribute((__nonnull__(1))); 
# 212
extern int strfromd(char * __dest, size_t __size, const char * __format, double __f) throw()
# 214
 __attribute((__nonnull__(3))); 
# 216
extern int strfromf(char * __dest, size_t __size, const char * __format, float __f) throw()
# 218
 __attribute((__nonnull__(3))); 
# 220
extern int strfroml(char * __dest, size_t __size, const char * __format, long double __f) throw()
# 222
 __attribute((__nonnull__(3))); 
# 232 "/usr/include/stdlib.h" 3
extern int strfromf32(char * __dest, size_t __size, const char * __format, _Float32 __f) throw()
# 234
 __attribute((__nonnull__(3))); 
# 238
extern int strfromf64(char * __dest, size_t __size, const char * __format, _Float64 __f) throw()
# 240
 __attribute((__nonnull__(3))); 
# 244
extern int strfromf128(char * __dest, size_t __size, const char * __format, _Float128 __f) throw()
# 246
 __attribute((__nonnull__(3))); 
# 250
extern int strfromf32x(char * __dest, size_t __size, const char * __format, _Float32x __f) throw()
# 252
 __attribute((__nonnull__(3))); 
# 256
extern int strfromf64x(char * __dest, size_t __size, const char * __format, _Float64x __f) throw()
# 258
 __attribute((__nonnull__(3))); 
# 274 "/usr/include/stdlib.h" 3
extern long strtol_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, locale_t __loc) throw()
# 276
 __attribute((__nonnull__(1, 4))); 
# 278
extern unsigned long strtoul_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, locale_t __loc) throw()
# 281
 __attribute((__nonnull__(1, 4))); 
# 284
__extension__ extern long long strtoll_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, locale_t __loc) throw()
# 287
 __attribute((__nonnull__(1, 4))); 
# 290
__extension__ extern unsigned long long strtoull_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, locale_t __loc) throw()
# 293
 __attribute((__nonnull__(1, 4))); 
# 295
extern double strtod_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, locale_t __loc) throw()
# 297
 __attribute((__nonnull__(1, 3))); 
# 299
extern float strtof_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, locale_t __loc) throw()
# 301
 __attribute((__nonnull__(1, 3))); 
# 303
extern long double strtold_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, locale_t __loc) throw()
# 306
 __attribute((__nonnull__(1, 3))); 
# 316 "/usr/include/stdlib.h" 3
extern _Float32 strtof32_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, locale_t __loc) throw()
# 319
 __attribute((__nonnull__(1, 3))); 
# 323
extern _Float64 strtof64_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, locale_t __loc) throw()
# 326
 __attribute((__nonnull__(1, 3))); 
# 330
extern _Float128 strtof128_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, locale_t __loc) throw()
# 333
 __attribute((__nonnull__(1, 3))); 
# 337
extern _Float32x strtof32x_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, locale_t __loc) throw()
# 340
 __attribute((__nonnull__(1, 3))); 
# 344
extern _Float64x strtof64x_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, locale_t __loc) throw()
# 347
 __attribute((__nonnull__(1, 3))); 
# 385 "/usr/include/stdlib.h" 3
extern char *l64a(long __n) throw(); 
# 388
extern long a64l(const char * __s) throw()
# 389
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 27 "/usr/include/sys/types.h" 3
extern "C" {
# 33
typedef __u_char u_char; 
# 34
typedef __u_short u_short; 
# 35
typedef __u_int u_int; 
# 36
typedef __u_long u_long; 
# 37
typedef __quad_t quad_t; 
# 38
typedef __u_quad_t u_quad_t; 
# 39
typedef __fsid_t fsid_t; 
# 42
typedef __loff_t loff_t; 
# 47
typedef __ino_t ino_t; 
# 54
typedef __ino64_t ino64_t; 
# 59
typedef __dev_t dev_t; 
# 64
typedef __gid_t gid_t; 
# 69
typedef __mode_t mode_t; 
# 74
typedef __nlink_t nlink_t; 
# 79
typedef __uid_t uid_t; 
# 85
typedef __off_t off_t; 
# 92
typedef __off64_t off64_t; 
# 103 "/usr/include/sys/types.h" 3
typedef __id_t id_t; 
# 108
typedef __ssize_t ssize_t; 
# 114
typedef __daddr_t daddr_t; 
# 115
typedef __caddr_t caddr_t; 
# 121
typedef __key_t key_t; 
# 134 "/usr/include/sys/types.h" 3
typedef __useconds_t useconds_t; 
# 138
typedef __suseconds_t suseconds_t; 
# 148 "/usr/include/sys/types.h" 3
typedef unsigned long ulong; 
# 149
typedef unsigned short ushort; 
# 150
typedef unsigned uint; 
# 24 "/usr/include/bits/stdint-intn.h" 3
typedef __int8_t int8_t; 
# 25
typedef __int16_t int16_t; 
# 26
typedef __int32_t int32_t; 
# 27
typedef __int64_t int64_t; 
# 158 "/usr/include/sys/types.h" 3
typedef __uint8_t u_int8_t; 
# 159
typedef __uint16_t u_int16_t; 
# 160
typedef __uint32_t u_int32_t; 
# 161
typedef __uint64_t u_int64_t; 
# 164
typedef long register_t __attribute((__mode__(__word__))); 
# 34 "/usr/include/bits/byteswap.h" 3
static inline __uint16_t __bswap_16(__uint16_t __bsx) 
# 35
{ 
# 37
return __builtin_bswap16(__bsx); 
# 41
} 
# 49
static inline __uint32_t __bswap_32(__uint32_t __bsx) 
# 50
{ 
# 52
return __builtin_bswap32(__bsx); 
# 56
} 
# 70 "/usr/include/bits/byteswap.h" 3
__extension__ static inline __uint64_t __bswap_64(__uint64_t __bsx) 
# 71
{ 
# 73
return __builtin_bswap64(__bsx); 
# 77
} 
# 33 "/usr/include/bits/uintn-identity.h" 3
static inline __uint16_t __uint16_identity(__uint16_t __x) 
# 34
{ 
# 35
return __x; 
# 36
} 
# 39
static inline __uint32_t __uint32_identity(__uint32_t __x) 
# 40
{ 
# 41
return __x; 
# 42
} 
# 45
static inline __uint64_t __uint64_identity(__uint64_t __x) 
# 46
{ 
# 47
return __x; 
# 48
} 
# 8 "/usr/include/bits/types/__sigset_t.h" 3
typedef 
# 6
struct { 
# 7
unsigned long __val[(1024) / ((8) * sizeof(unsigned long))]; 
# 8
} __sigset_t; 
# 7 "/usr/include/bits/types/sigset_t.h" 3
typedef __sigset_t sigset_t; 
# 49 "/usr/include/sys/select.h" 3
typedef long __fd_mask; 
# 70 "/usr/include/sys/select.h" 3
typedef 
# 60
struct { 
# 64
__fd_mask fds_bits[1024 / (8 * ((int)sizeof(__fd_mask)))]; 
# 70
} fd_set; 
# 77
typedef __fd_mask fd_mask; 
# 91 "/usr/include/sys/select.h" 3
extern "C" {
# 101 "/usr/include/sys/select.h" 3
extern int select(int __nfds, fd_set *__restrict__ __readfds, fd_set *__restrict__ __writefds, fd_set *__restrict__ __exceptfds, timeval *__restrict__ __timeout); 
# 113 "/usr/include/sys/select.h" 3
extern int pselect(int __nfds, fd_set *__restrict__ __readfds, fd_set *__restrict__ __writefds, fd_set *__restrict__ __exceptfds, const timespec *__restrict__ __timeout, const __sigset_t *__restrict__ __sigmask); 
# 126 "/usr/include/sys/select.h" 3
}
# 185 "/usr/include/sys/types.h" 3
typedef __blksize_t blksize_t; 
# 192
typedef __blkcnt_t blkcnt_t; 
# 196
typedef __fsblkcnt_t fsblkcnt_t; 
# 200
typedef __fsfilcnt_t fsfilcnt_t; 
# 219 "/usr/include/sys/types.h" 3
typedef __blkcnt64_t blkcnt64_t; 
# 220
typedef __fsblkcnt64_t fsblkcnt64_t; 
# 221
typedef __fsfilcnt64_t fsfilcnt64_t; 
# 65 "/usr/include/bits/pthreadtypes-arch.h" 3
struct __pthread_rwlock_arch_t { 
# 67
unsigned __readers; 
# 68
unsigned __writers; 
# 69
unsigned __wrphase_futex; 
# 70
unsigned __writers_futex; 
# 71
unsigned __pad3; 
# 72
unsigned __pad4; 
# 74
int __cur_writer; 
# 75
int __shared; 
# 76
signed char __rwelision; 
# 81
unsigned char __pad1[7]; 
# 84
unsigned long __pad2; 
# 87
unsigned __flags; 
# 99 "/usr/include/bits/pthreadtypes-arch.h" 3
}; 
# 86 "/usr/include/bits/thread-shared-types.h" 3
typedef 
# 82
struct __pthread_internal_list { 
# 84
__pthread_internal_list *__prev; 
# 85
__pthread_internal_list *__next; 
# 86
} __pthread_list_t; 
# 118 "/usr/include/bits/thread-shared-types.h" 3
struct __pthread_mutex_s { 
# 120
int __lock; 
# 121
unsigned __count; 
# 122
int __owner; 
# 124
unsigned __nusers; 
# 148 "/usr/include/bits/thread-shared-types.h" 3
int __kind; 
# 154
short __spins; short __elision; 
# 155
__pthread_list_t __list; 
# 166 "/usr/include/bits/thread-shared-types.h" 3
}; 
# 171
struct __pthread_cond_s { 
# 174
__extension__ union { 
# 175
__extension__ unsigned long long __wseq; 
# 177
struct { 
# 178
unsigned __low; 
# 179
unsigned __high; 
# 180
} __wseq32; 
# 181
}; 
# 183
__extension__ union { 
# 184
__extension__ unsigned long long __g1_start; 
# 186
struct { 
# 187
unsigned __low; 
# 188
unsigned __high; 
# 189
} __g1_start32; 
# 190
}; 
# 191
unsigned __g_refs[2]; 
# 192
unsigned __g_size[2]; 
# 193
unsigned __g1_orig_size; 
# 194
unsigned __wrefs; 
# 195
unsigned __g_signals[2]; 
# 196
}; 
# 27 "/usr/include/bits/pthreadtypes.h" 3
typedef unsigned long pthread_t; 
# 36
typedef 
# 33
union { 
# 34
char __size[4]; 
# 35
int __align; 
# 36
} pthread_mutexattr_t; 
# 45
typedef 
# 42
union { 
# 43
char __size[4]; 
# 44
int __align; 
# 45
} pthread_condattr_t; 
# 49
typedef unsigned pthread_key_t; 
# 53
typedef int pthread_once_t; 
# 56
union pthread_attr_t { 
# 58
char __size[56]; 
# 59
long __align; 
# 60
}; 
# 62
typedef pthread_attr_t pthread_attr_t; 
# 72
typedef 
# 68
union { 
# 69
__pthread_mutex_s __data; 
# 70
char __size[40]; 
# 71
long __align; 
# 72
} pthread_mutex_t; 
# 80
typedef 
# 76
union { 
# 77
__pthread_cond_s __data; 
# 78
char __size[48]; 
# 79
__extension__ long long __align; 
# 80
} pthread_cond_t; 
# 91
typedef 
# 87
union { 
# 88
__pthread_rwlock_arch_t __data; 
# 89
char __size[56]; 
# 90
long __align; 
# 91
} pthread_rwlock_t; 
# 97
typedef 
# 94
union { 
# 95
char __size[8]; 
# 96
long __align; 
# 97
} pthread_rwlockattr_t; 
# 103
typedef volatile int pthread_spinlock_t; 
# 112
typedef 
# 109
union { 
# 110
char __size[32]; 
# 111
long __align; 
# 112
} pthread_barrier_t; 
# 118
typedef 
# 115
union { 
# 116
char __size[4]; 
# 117
int __align; 
# 118
} pthread_barrierattr_t; 
# 230 "/usr/include/sys/types.h" 3
}
# 401 "/usr/include/stdlib.h" 3
extern long random() throw(); 
# 404
extern void srandom(unsigned __seed) throw(); 
# 410
extern char *initstate(unsigned __seed, char * __statebuf, size_t __statelen) throw()
# 411
 __attribute((__nonnull__(2))); 
# 415
extern char *setstate(char * __statebuf) throw() __attribute((__nonnull__(1))); 
# 423
struct random_data { 
# 425
int32_t *fptr; 
# 426
int32_t *rptr; 
# 427
int32_t *state; 
# 428
int rand_type; 
# 429
int rand_deg; 
# 430
int rand_sep; 
# 431
int32_t *end_ptr; 
# 432
}; 
# 434
extern int random_r(random_data *__restrict__ __buf, int32_t *__restrict__ __result) throw()
# 435
 __attribute((__nonnull__(1, 2))); 
# 437
extern int srandom_r(unsigned __seed, random_data * __buf) throw()
# 438
 __attribute((__nonnull__(2))); 
# 440
extern int initstate_r(unsigned __seed, char *__restrict__ __statebuf, size_t __statelen, random_data *__restrict__ __buf) throw()
# 443
 __attribute((__nonnull__(2, 4))); 
# 445
extern int setstate_r(char *__restrict__ __statebuf, random_data *__restrict__ __buf) throw()
# 447
 __attribute((__nonnull__(1, 2))); 
# 453
extern int rand() throw(); 
# 455
extern void srand(unsigned __seed) throw(); 
# 459
extern int rand_r(unsigned * __seed) throw(); 
# 467
extern double drand48() throw(); 
# 468
extern double erand48(unsigned short  __xsubi[3]) throw() __attribute((__nonnull__(1))); 
# 471
extern long lrand48() throw(); 
# 472
extern long nrand48(unsigned short  __xsubi[3]) throw()
# 473
 __attribute((__nonnull__(1))); 
# 476
extern long mrand48() throw(); 
# 477
extern long jrand48(unsigned short  __xsubi[3]) throw()
# 478
 __attribute((__nonnull__(1))); 
# 481
extern void srand48(long __seedval) throw(); 
# 482
extern unsigned short *seed48(unsigned short  __seed16v[3]) throw()
# 483
 __attribute((__nonnull__(1))); 
# 484
extern void lcong48(unsigned short  __param[7]) throw() __attribute((__nonnull__(1))); 
# 490
struct drand48_data { 
# 492
unsigned short __x[3]; 
# 493
unsigned short __old_x[3]; 
# 494
unsigned short __c; 
# 495
unsigned short __init; 
# 496
__extension__ unsigned long long __a; 
# 498
}; 
# 501
extern int drand48_r(drand48_data *__restrict__ __buffer, double *__restrict__ __result) throw()
# 502
 __attribute((__nonnull__(1, 2))); 
# 503
extern int erand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, double *__restrict__ __result) throw()
# 505
 __attribute((__nonnull__(1, 2))); 
# 508
extern int lrand48_r(drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 510
 __attribute((__nonnull__(1, 2))); 
# 511
extern int nrand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 514
 __attribute((__nonnull__(1, 2))); 
# 517
extern int mrand48_r(drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 519
 __attribute((__nonnull__(1, 2))); 
# 520
extern int jrand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 523
 __attribute((__nonnull__(1, 2))); 
# 526
extern int srand48_r(long __seedval, drand48_data * __buffer) throw()
# 527
 __attribute((__nonnull__(2))); 
# 529
extern int seed48_r(unsigned short  __seed16v[3], drand48_data * __buffer) throw()
# 530
 __attribute((__nonnull__(1, 2))); 
# 532
extern int lcong48_r(unsigned short  __param[7], drand48_data * __buffer) throw()
# 534
 __attribute((__nonnull__(1, 2))); 
# 539
extern void *malloc(size_t __size) throw() __attribute((__malloc__)); 
# 541
extern void *calloc(size_t __nmemb, size_t __size) throw()
# 542
 __attribute((__malloc__)); 
# 549
extern void *realloc(void * __ptr, size_t __size) throw()
# 550
 __attribute((__warn_unused_result__)); 
# 558
extern void *reallocarray(void * __ptr, size_t __nmemb, size_t __size) throw()
# 559
 __attribute((__warn_unused_result__)); 
# 563
extern void free(void * __ptr) throw(); 
# 26 "/usr/include/alloca.h" 3
extern "C" {
# 32
extern void *alloca(size_t __size) throw(); 
# 38
}
# 572 "/usr/include/stdlib.h" 3
extern void *valloc(size_t __size) throw() __attribute((__malloc__)); 
# 577
extern int posix_memalign(void ** __memptr, size_t __alignment, size_t __size) throw()
# 578
 __attribute((__nonnull__(1))); 
# 583
extern void *aligned_alloc(size_t __alignment, size_t __size) throw()
# 584
 __attribute((__malloc__)) __attribute((__alloc_size__(2))); 
# 588
extern void abort() throw() __attribute((__noreturn__)); 
# 592
extern int atexit(void (* __func)(void)) throw() __attribute((__nonnull__(1))); 
# 597
extern "C++" int at_quick_exit(void (* __func)(void)) throw() __asm__("at_quick_exit")
# 598
 __attribute((__nonnull__(1))); 
# 607 "/usr/include/stdlib.h" 3
extern int on_exit(void (* __func)(int __status, void * __arg), void * __arg) throw()
# 608
 __attribute((__nonnull__(1))); 
# 614
extern void exit(int __status) throw() __attribute((__noreturn__)); 
# 620
extern void quick_exit(int __status) throw() __attribute((__noreturn__)); 
# 626
extern void _Exit(int __status) throw() __attribute((__noreturn__)); 
# 631
extern char *getenv(const char * __name) throw() __attribute((__nonnull__(1))); 
# 636
extern char *secure_getenv(const char * __name) throw()
# 637
 __attribute((__nonnull__(1))); 
# 644
extern int putenv(char * __string) throw() __attribute((__nonnull__(1))); 
# 650
extern int setenv(const char * __name, const char * __value, int __replace) throw()
# 651
 __attribute((__nonnull__(2))); 
# 654
extern int unsetenv(const char * __name) throw() __attribute((__nonnull__(1))); 
# 661
extern int clearenv() throw(); 
# 672 "/usr/include/stdlib.h" 3
extern char *mktemp(char * __template) throw() __attribute((__nonnull__(1))); 
# 685 "/usr/include/stdlib.h" 3
extern int mkstemp(char * __template) __attribute((__nonnull__(1))); 
# 695 "/usr/include/stdlib.h" 3
extern int mkstemp64(char * __template) __attribute((__nonnull__(1))); 
# 707 "/usr/include/stdlib.h" 3
extern int mkstemps(char * __template, int __suffixlen) __attribute((__nonnull__(1))); 
# 717 "/usr/include/stdlib.h" 3
extern int mkstemps64(char * __template, int __suffixlen)
# 718
 __attribute((__nonnull__(1))); 
# 728 "/usr/include/stdlib.h" 3
extern char *mkdtemp(char * __template) throw() __attribute((__nonnull__(1))); 
# 739 "/usr/include/stdlib.h" 3
extern int mkostemp(char * __template, int __flags) __attribute((__nonnull__(1))); 
# 749 "/usr/include/stdlib.h" 3
extern int mkostemp64(char * __template, int __flags) __attribute((__nonnull__(1))); 
# 759 "/usr/include/stdlib.h" 3
extern int mkostemps(char * __template, int __suffixlen, int __flags)
# 760
 __attribute((__nonnull__(1))); 
# 771 "/usr/include/stdlib.h" 3
extern int mkostemps64(char * __template, int __suffixlen, int __flags)
# 772
 __attribute((__nonnull__(1))); 
# 781 "/usr/include/stdlib.h" 3
extern int system(const char * __command); 
# 787
extern char *canonicalize_file_name(const char * __name) throw()
# 788
 __attribute((__nonnull__(1))); 
# 797 "/usr/include/stdlib.h" 3
extern char *realpath(const char *__restrict__ __name, char *__restrict__ __resolved) throw(); 
# 805
typedef int (*__compar_fn_t)(const void *, const void *); 
# 808
typedef __compar_fn_t comparison_fn_t; 
# 812
typedef int (*__compar_d_fn_t)(const void *, const void *, void *); 
# 817
extern void *bsearch(const void * __key, const void * __base, size_t __nmemb, size_t __size, __compar_fn_t __compar)
# 819
 __attribute((__nonnull__(1, 2, 5))); 
# 827
extern void qsort(void * __base, size_t __nmemb, size_t __size, __compar_fn_t __compar)
# 828
 __attribute((__nonnull__(1, 4))); 
# 830
extern void qsort_r(void * __base, size_t __nmemb, size_t __size, __compar_d_fn_t __compar, void * __arg)
# 832
 __attribute((__nonnull__(1, 4))); 
# 837
extern int abs(int __x) throw() __attribute((const)); 
# 838
extern long labs(long __x) throw() __attribute((const)); 
# 841
__extension__ extern long long llabs(long long __x) throw()
# 842
 __attribute((const)); 
# 849
extern div_t div(int __numer, int __denom) throw()
# 850
 __attribute((const)); 
# 851
extern ldiv_t ldiv(long __numer, long __denom) throw()
# 852
 __attribute((const)); 
# 855
__extension__ extern lldiv_t lldiv(long long __numer, long long __denom) throw()
# 857
 __attribute((const)); 
# 869 "/usr/include/stdlib.h" 3
extern char *ecvt(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 870
 __attribute((__nonnull__(3, 4))); 
# 875
extern char *fcvt(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 876
 __attribute((__nonnull__(3, 4))); 
# 881
extern char *gcvt(double __value, int __ndigit, char * __buf) throw()
# 882
 __attribute((__nonnull__(3))); 
# 887
extern char *qecvt(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 889
 __attribute((__nonnull__(3, 4))); 
# 890
extern char *qfcvt(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 892
 __attribute((__nonnull__(3, 4))); 
# 893
extern char *qgcvt(long double __value, int __ndigit, char * __buf) throw()
# 894
 __attribute((__nonnull__(3))); 
# 899
extern int ecvt_r(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 901
 __attribute((__nonnull__(3, 4, 5))); 
# 902
extern int fcvt_r(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 904
 __attribute((__nonnull__(3, 4, 5))); 
# 906
extern int qecvt_r(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 909
 __attribute((__nonnull__(3, 4, 5))); 
# 910
extern int qfcvt_r(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 913
 __attribute((__nonnull__(3, 4, 5))); 
# 919
extern int mblen(const char * __s, size_t __n) throw(); 
# 922
extern int mbtowc(wchar_t *__restrict__ __pwc, const char *__restrict__ __s, size_t __n) throw(); 
# 926
extern int wctomb(char * __s, wchar_t __wchar) throw(); 
# 930
extern size_t mbstowcs(wchar_t *__restrict__ __pwcs, const char *__restrict__ __s, size_t __n) throw(); 
# 933
extern size_t wcstombs(char *__restrict__ __s, const wchar_t *__restrict__ __pwcs, size_t __n) throw(); 
# 943
extern int rpmatch(const char * __response) throw() __attribute((__nonnull__(1))); 
# 954 "/usr/include/stdlib.h" 3
extern int getsubopt(char **__restrict__ __optionp, char *const *__restrict__ __tokens, char **__restrict__ __valuep) throw()
# 957
 __attribute((__nonnull__(1, 2, 3))); 
# 965
extern int posix_openpt(int __oflag); 
# 973
extern int grantpt(int __fd) throw(); 
# 977
extern int unlockpt(int __fd) throw(); 
# 982
extern char *ptsname(int __fd) throw(); 
# 989
extern int ptsname_r(int __fd, char * __buf, size_t __buflen) throw()
# 990
 __attribute((__nonnull__(2))); 
# 993
extern int getpt(); 
# 1000
extern int getloadavg(double  __loadavg[], int __nelem) throw()
# 1001
 __attribute((__nonnull__(1))); 
# 1020 "/usr/include/stdlib.h" 3
}
# 46 "/usr/include/c++/8/bits/std_abs.h" 3
extern "C++" {
# 48
namespace std __attribute((__visibility__("default"))) { 
# 52
using ::abs;
# 56
inline long abs(long __i) { return __builtin_labs(__i); } 
# 61
inline long long abs(long long __x) { return __builtin_llabs(__x); } 
# 70
constexpr double abs(double __x) 
# 71
{ return __builtin_fabs(__x); } 
# 74
constexpr float abs(float __x) 
# 75
{ return __builtin_fabsf(__x); } 
# 78
constexpr long double abs(long double __x) 
# 79
{ return __builtin_fabsl(__x); } 
# 84
constexpr __int128 abs(__int128 __x) { return (__x >= (0)) ? __x : (-__x); } 
# 102 "/usr/include/c++/8/bits/std_abs.h" 3
constexpr __float128 abs(__float128 __x) 
# 103
{ return (__x < (0)) ? -__x : __x; } 
# 107
}
# 108
}
# 77 "/usr/include/c++/8/cmath" 3
extern "C++" {
# 79
namespace std __attribute((__visibility__("default"))) { 
# 83
using ::acos;
# 87
constexpr float acos(float __x) 
# 88
{ return __builtin_acosf(__x); } 
# 91
constexpr long double acos(long double __x) 
# 92
{ return __builtin_acosl(__x); } 
# 95
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 99
acos(_Tp __x) 
# 100
{ return __builtin_acos(__x); } 
# 102
using ::asin;
# 106
constexpr float asin(float __x) 
# 107
{ return __builtin_asinf(__x); } 
# 110
constexpr long double asin(long double __x) 
# 111
{ return __builtin_asinl(__x); } 
# 114
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 118
asin(_Tp __x) 
# 119
{ return __builtin_asin(__x); } 
# 121
using ::atan;
# 125
constexpr float atan(float __x) 
# 126
{ return __builtin_atanf(__x); } 
# 129
constexpr long double atan(long double __x) 
# 130
{ return __builtin_atanl(__x); } 
# 133
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 137
atan(_Tp __x) 
# 138
{ return __builtin_atan(__x); } 
# 140
using ::atan2;
# 144
constexpr float atan2(float __y, float __x) 
# 145
{ return __builtin_atan2f(__y, __x); } 
# 148
constexpr long double atan2(long double __y, long double __x) 
# 149
{ return __builtin_atan2l(__y, __x); } 
# 152
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 155
atan2(_Tp __y, _Up __x) 
# 156
{ 
# 157
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 158
return atan2((__type)__y, (__type)__x); 
# 159
} 
# 161
using ::ceil;
# 165
constexpr float ceil(float __x) 
# 166
{ return __builtin_ceilf(__x); } 
# 169
constexpr long double ceil(long double __x) 
# 170
{ return __builtin_ceill(__x); } 
# 173
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 177
ceil(_Tp __x) 
# 178
{ return __builtin_ceil(__x); } 
# 180
using ::cos;
# 184
constexpr float cos(float __x) 
# 185
{ return __builtin_cosf(__x); } 
# 188
constexpr long double cos(long double __x) 
# 189
{ return __builtin_cosl(__x); } 
# 192
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 196
cos(_Tp __x) 
# 197
{ return __builtin_cos(__x); } 
# 199
using ::cosh;
# 203
constexpr float cosh(float __x) 
# 204
{ return __builtin_coshf(__x); } 
# 207
constexpr long double cosh(long double __x) 
# 208
{ return __builtin_coshl(__x); } 
# 211
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 215
cosh(_Tp __x) 
# 216
{ return __builtin_cosh(__x); } 
# 218
using ::exp;
# 222
constexpr float exp(float __x) 
# 223
{ return __builtin_expf(__x); } 
# 226
constexpr long double exp(long double __x) 
# 227
{ return __builtin_expl(__x); } 
# 230
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 234
exp(_Tp __x) 
# 235
{ return __builtin_exp(__x); } 
# 237
using ::fabs;
# 241
constexpr float fabs(float __x) 
# 242
{ return __builtin_fabsf(__x); } 
# 245
constexpr long double fabs(long double __x) 
# 246
{ return __builtin_fabsl(__x); } 
# 249
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 253
fabs(_Tp __x) 
# 254
{ return __builtin_fabs(__x); } 
# 256
using ::floor;
# 260
constexpr float floor(float __x) 
# 261
{ return __builtin_floorf(__x); } 
# 264
constexpr long double floor(long double __x) 
# 265
{ return __builtin_floorl(__x); } 
# 268
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 272
floor(_Tp __x) 
# 273
{ return __builtin_floor(__x); } 
# 275
using ::fmod;
# 279
constexpr float fmod(float __x, float __y) 
# 280
{ return __builtin_fmodf(__x, __y); } 
# 283
constexpr long double fmod(long double __x, long double __y) 
# 284
{ return __builtin_fmodl(__x, __y); } 
# 287
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 290
fmod(_Tp __x, _Up __y) 
# 291
{ 
# 292
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 293
return fmod((__type)__x, (__type)__y); 
# 294
} 
# 296
using ::frexp;
# 300
inline float frexp(float __x, int *__exp) 
# 301
{ return __builtin_frexpf(__x, __exp); } 
# 304
inline long double frexp(long double __x, int *__exp) 
# 305
{ return __builtin_frexpl(__x, __exp); } 
# 308
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 312
frexp(_Tp __x, int *__exp) 
# 313
{ return __builtin_frexp(__x, __exp); } 
# 315
using ::ldexp;
# 319
constexpr float ldexp(float __x, int __exp) 
# 320
{ return __builtin_ldexpf(__x, __exp); } 
# 323
constexpr long double ldexp(long double __x, int __exp) 
# 324
{ return __builtin_ldexpl(__x, __exp); } 
# 327
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 331
ldexp(_Tp __x, int __exp) 
# 332
{ return __builtin_ldexp(__x, __exp); } 
# 334
using ::log;
# 338
constexpr float log(float __x) 
# 339
{ return __builtin_logf(__x); } 
# 342
constexpr long double log(long double __x) 
# 343
{ return __builtin_logl(__x); } 
# 346
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 350
log(_Tp __x) 
# 351
{ return __builtin_log(__x); } 
# 353
using ::log10;
# 357
constexpr float log10(float __x) 
# 358
{ return __builtin_log10f(__x); } 
# 361
constexpr long double log10(long double __x) 
# 362
{ return __builtin_log10l(__x); } 
# 365
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 369
log10(_Tp __x) 
# 370
{ return __builtin_log10(__x); } 
# 372
using ::modf;
# 376
inline float modf(float __x, float *__iptr) 
# 377
{ return __builtin_modff(__x, __iptr); } 
# 380
inline long double modf(long double __x, long double *__iptr) 
# 381
{ return __builtin_modfl(__x, __iptr); } 
# 384
using ::pow;
# 388
constexpr float pow(float __x, float __y) 
# 389
{ return __builtin_powf(__x, __y); } 
# 392
constexpr long double pow(long double __x, long double __y) 
# 393
{ return __builtin_powl(__x, __y); } 
# 412 "/usr/include/c++/8/cmath" 3
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 415
pow(_Tp __x, _Up __y) 
# 416
{ 
# 417
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 418
return pow((__type)__x, (__type)__y); 
# 419
} 
# 421
using ::sin;
# 425
constexpr float sin(float __x) 
# 426
{ return __builtin_sinf(__x); } 
# 429
constexpr long double sin(long double __x) 
# 430
{ return __builtin_sinl(__x); } 
# 433
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 437
sin(_Tp __x) 
# 438
{ return __builtin_sin(__x); } 
# 440
using ::sinh;
# 444
constexpr float sinh(float __x) 
# 445
{ return __builtin_sinhf(__x); } 
# 448
constexpr long double sinh(long double __x) 
# 449
{ return __builtin_sinhl(__x); } 
# 452
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 456
sinh(_Tp __x) 
# 457
{ return __builtin_sinh(__x); } 
# 459
using ::sqrt;
# 463
constexpr float sqrt(float __x) 
# 464
{ return __builtin_sqrtf(__x); } 
# 467
constexpr long double sqrt(long double __x) 
# 468
{ return __builtin_sqrtl(__x); } 
# 471
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 475
sqrt(_Tp __x) 
# 476
{ return __builtin_sqrt(__x); } 
# 478
using ::tan;
# 482
constexpr float tan(float __x) 
# 483
{ return __builtin_tanf(__x); } 
# 486
constexpr long double tan(long double __x) 
# 487
{ return __builtin_tanl(__x); } 
# 490
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 494
tan(_Tp __x) 
# 495
{ return __builtin_tan(__x); } 
# 497
using ::tanh;
# 501
constexpr float tanh(float __x) 
# 502
{ return __builtin_tanhf(__x); } 
# 505
constexpr long double tanh(long double __x) 
# 506
{ return __builtin_tanhl(__x); } 
# 509
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 513
tanh(_Tp __x) 
# 514
{ return __builtin_tanh(__x); } 
# 537 "/usr/include/c++/8/cmath" 3
constexpr int fpclassify(float __x) 
# 538
{ return __builtin_fpclassify(0, 1, 4, 3, 2, __x); 
# 539
} 
# 542
constexpr int fpclassify(double __x) 
# 543
{ return __builtin_fpclassify(0, 1, 4, 3, 2, __x); 
# 544
} 
# 547
constexpr int fpclassify(long double __x) 
# 548
{ return __builtin_fpclassify(0, 1, 4, 3, 2, __x); 
# 549
} 
# 553
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, int> ::__type 
# 556
fpclassify(_Tp __x) 
# 557
{ return (__x != 0) ? 4 : 2; } 
# 562
constexpr bool isfinite(float __x) 
# 563
{ return __builtin_isfinite(__x); } 
# 566
constexpr bool isfinite(double __x) 
# 567
{ return __builtin_isfinite(__x); } 
# 570
constexpr bool isfinite(long double __x) 
# 571
{ return __builtin_isfinite(__x); } 
# 575
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 578
isfinite(_Tp __x) 
# 579
{ return true; } 
# 584
constexpr bool isinf(float __x) 
# 585
{ return __builtin_isinf(__x); } 
# 592
constexpr bool isinf(double __x) 
# 593
{ return __builtin_isinf(__x); } 
# 597
constexpr bool isinf(long double __x) 
# 598
{ return __builtin_isinf(__x); } 
# 602
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 605
isinf(_Tp __x) 
# 606
{ return false; } 
# 611
constexpr bool isnan(float __x) 
# 612
{ return __builtin_isnan(__x); } 
# 619
constexpr bool isnan(double __x) 
# 620
{ return __builtin_isnan(__x); } 
# 624
constexpr bool isnan(long double __x) 
# 625
{ return __builtin_isnan(__x); } 
# 629
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 632
isnan(_Tp __x) 
# 633
{ return false; } 
# 638
constexpr bool isnormal(float __x) 
# 639
{ return __builtin_isnormal(__x); } 
# 642
constexpr bool isnormal(double __x) 
# 643
{ return __builtin_isnormal(__x); } 
# 646
constexpr bool isnormal(long double __x) 
# 647
{ return __builtin_isnormal(__x); } 
# 651
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 654
isnormal(_Tp __x) 
# 655
{ return (__x != 0) ? true : false; } 
# 661
constexpr bool signbit(float __x) 
# 662
{ return __builtin_signbit(__x); } 
# 665
constexpr bool signbit(double __x) 
# 666
{ return __builtin_signbit(__x); } 
# 669
constexpr bool signbit(long double __x) 
# 670
{ return __builtin_signbit(__x); } 
# 674
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 677
signbit(_Tp __x) 
# 678
{ return (__x < 0) ? true : false; } 
# 683
constexpr bool isgreater(float __x, float __y) 
# 684
{ return __builtin_isgreater(__x, __y); } 
# 687
constexpr bool isgreater(double __x, double __y) 
# 688
{ return __builtin_isgreater(__x, __y); } 
# 691
constexpr bool isgreater(long double __x, long double __y) 
# 692
{ return __builtin_isgreater(__x, __y); } 
# 696
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 700
isgreater(_Tp __x, _Up __y) 
# 701
{ 
# 702
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 703
return __builtin_isgreater((__type)__x, (__type)__y); 
# 704
} 
# 709
constexpr bool isgreaterequal(float __x, float __y) 
# 710
{ return __builtin_isgreaterequal(__x, __y); } 
# 713
constexpr bool isgreaterequal(double __x, double __y) 
# 714
{ return __builtin_isgreaterequal(__x, __y); } 
# 717
constexpr bool isgreaterequal(long double __x, long double __y) 
# 718
{ return __builtin_isgreaterequal(__x, __y); } 
# 722
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 726
isgreaterequal(_Tp __x, _Up __y) 
# 727
{ 
# 728
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 729
return __builtin_isgreaterequal((__type)__x, (__type)__y); 
# 730
} 
# 735
constexpr bool isless(float __x, float __y) 
# 736
{ return __builtin_isless(__x, __y); } 
# 739
constexpr bool isless(double __x, double __y) 
# 740
{ return __builtin_isless(__x, __y); } 
# 743
constexpr bool isless(long double __x, long double __y) 
# 744
{ return __builtin_isless(__x, __y); } 
# 748
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 752
isless(_Tp __x, _Up __y) 
# 753
{ 
# 754
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 755
return __builtin_isless((__type)__x, (__type)__y); 
# 756
} 
# 761
constexpr bool islessequal(float __x, float __y) 
# 762
{ return __builtin_islessequal(__x, __y); } 
# 765
constexpr bool islessequal(double __x, double __y) 
# 766
{ return __builtin_islessequal(__x, __y); } 
# 769
constexpr bool islessequal(long double __x, long double __y) 
# 770
{ return __builtin_islessequal(__x, __y); } 
# 774
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 778
islessequal(_Tp __x, _Up __y) 
# 779
{ 
# 780
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 781
return __builtin_islessequal((__type)__x, (__type)__y); 
# 782
} 
# 787
constexpr bool islessgreater(float __x, float __y) 
# 788
{ return __builtin_islessgreater(__x, __y); } 
# 791
constexpr bool islessgreater(double __x, double __y) 
# 792
{ return __builtin_islessgreater(__x, __y); } 
# 795
constexpr bool islessgreater(long double __x, long double __y) 
# 796
{ return __builtin_islessgreater(__x, __y); } 
# 800
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 804
islessgreater(_Tp __x, _Up __y) 
# 805
{ 
# 806
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 807
return __builtin_islessgreater((__type)__x, (__type)__y); 
# 808
} 
# 813
constexpr bool isunordered(float __x, float __y) 
# 814
{ return __builtin_isunordered(__x, __y); } 
# 817
constexpr bool isunordered(double __x, double __y) 
# 818
{ return __builtin_isunordered(__x, __y); } 
# 821
constexpr bool isunordered(long double __x, long double __y) 
# 822
{ return __builtin_isunordered(__x, __y); } 
# 826
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 830
isunordered(_Tp __x, _Up __y) 
# 831
{ 
# 832
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 833
return __builtin_isunordered((__type)__x, (__type)__y); 
# 834
} 
# 1065 "/usr/include/c++/8/cmath" 3
using ::double_t;
# 1066
using ::float_t;
# 1069
using ::acosh;
# 1070
using ::acoshf;
# 1071
using ::acoshl;
# 1073
using ::asinh;
# 1074
using ::asinhf;
# 1075
using ::asinhl;
# 1077
using ::atanh;
# 1078
using ::atanhf;
# 1079
using ::atanhl;
# 1081
using ::cbrt;
# 1082
using ::cbrtf;
# 1083
using ::cbrtl;
# 1085
using ::copysign;
# 1086
using ::copysignf;
# 1087
using ::copysignl;
# 1089
using ::erf;
# 1090
using ::erff;
# 1091
using ::erfl;
# 1093
using ::erfc;
# 1094
using ::erfcf;
# 1095
using ::erfcl;
# 1097
using ::exp2;
# 1098
using ::exp2f;
# 1099
using ::exp2l;
# 1101
using ::expm1;
# 1102
using ::expm1f;
# 1103
using ::expm1l;
# 1105
using ::fdim;
# 1106
using ::fdimf;
# 1107
using ::fdiml;
# 1109
using ::fma;
# 1110
using ::fmaf;
# 1111
using ::fmal;
# 1113
using ::fmax;
# 1114
using ::fmaxf;
# 1115
using ::fmaxl;
# 1117
using ::fmin;
# 1118
using ::fminf;
# 1119
using ::fminl;
# 1121
using ::hypot;
# 1122
using ::hypotf;
# 1123
using ::hypotl;
# 1125
using ::ilogb;
# 1126
using ::ilogbf;
# 1127
using ::ilogbl;
# 1129
using ::lgamma;
# 1130
using ::lgammaf;
# 1131
using ::lgammal;
# 1134
using ::llrint;
# 1135
using ::llrintf;
# 1136
using ::llrintl;
# 1138
using ::llround;
# 1139
using ::llroundf;
# 1140
using ::llroundl;
# 1143
using ::log1p;
# 1144
using ::log1pf;
# 1145
using ::log1pl;
# 1147
using ::log2;
# 1148
using ::log2f;
# 1149
using ::log2l;
# 1151
using ::logb;
# 1152
using ::logbf;
# 1153
using ::logbl;
# 1155
using ::lrint;
# 1156
using ::lrintf;
# 1157
using ::lrintl;
# 1159
using ::lround;
# 1160
using ::lroundf;
# 1161
using ::lroundl;
# 1163
using ::nan;
# 1164
using ::nanf;
# 1165
using ::nanl;
# 1167
using ::nearbyint;
# 1168
using ::nearbyintf;
# 1169
using ::nearbyintl;
# 1171
using ::nextafter;
# 1172
using ::nextafterf;
# 1173
using ::nextafterl;
# 1175
using ::nexttoward;
# 1176
using ::nexttowardf;
# 1177
using ::nexttowardl;
# 1179
using ::remainder;
# 1180
using ::remainderf;
# 1181
using ::remainderl;
# 1183
using ::remquo;
# 1184
using ::remquof;
# 1185
using ::remquol;
# 1187
using ::rint;
# 1188
using ::rintf;
# 1189
using ::rintl;
# 1191
using ::round;
# 1192
using ::roundf;
# 1193
using ::roundl;
# 1195
using ::scalbln;
# 1196
using ::scalblnf;
# 1197
using ::scalblnl;
# 1199
using ::scalbn;
# 1200
using ::scalbnf;
# 1201
using ::scalbnl;
# 1203
using ::tgamma;
# 1204
using ::tgammaf;
# 1205
using ::tgammal;
# 1207
using ::trunc;
# 1208
using ::truncf;
# 1209
using ::truncl;
# 1214
constexpr float acosh(float __x) 
# 1215
{ return __builtin_acoshf(__x); } 
# 1218
constexpr long double acosh(long double __x) 
# 1219
{ return __builtin_acoshl(__x); } 
# 1223
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1226
acosh(_Tp __x) 
# 1227
{ return __builtin_acosh(__x); } 
# 1232
constexpr float asinh(float __x) 
# 1233
{ return __builtin_asinhf(__x); } 
# 1236
constexpr long double asinh(long double __x) 
# 1237
{ return __builtin_asinhl(__x); } 
# 1241
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1244
asinh(_Tp __x) 
# 1245
{ return __builtin_asinh(__x); } 
# 1250
constexpr float atanh(float __x) 
# 1251
{ return __builtin_atanhf(__x); } 
# 1254
constexpr long double atanh(long double __x) 
# 1255
{ return __builtin_atanhl(__x); } 
# 1259
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1262
atanh(_Tp __x) 
# 1263
{ return __builtin_atanh(__x); } 
# 1268
constexpr float cbrt(float __x) 
# 1269
{ return __builtin_cbrtf(__x); } 
# 1272
constexpr long double cbrt(long double __x) 
# 1273
{ return __builtin_cbrtl(__x); } 
# 1277
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1280
cbrt(_Tp __x) 
# 1281
{ return __builtin_cbrt(__x); } 
# 1286
constexpr float copysign(float __x, float __y) 
# 1287
{ return __builtin_copysignf(__x, __y); } 
# 1290
constexpr long double copysign(long double __x, long double __y) 
# 1291
{ return __builtin_copysignl(__x, __y); } 
# 1295
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1297
copysign(_Tp __x, _Up __y) 
# 1298
{ 
# 1299
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1300
return copysign((__type)__x, (__type)__y); 
# 1301
} 
# 1306
constexpr float erf(float __x) 
# 1307
{ return __builtin_erff(__x); } 
# 1310
constexpr long double erf(long double __x) 
# 1311
{ return __builtin_erfl(__x); } 
# 1315
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1318
erf(_Tp __x) 
# 1319
{ return __builtin_erf(__x); } 
# 1324
constexpr float erfc(float __x) 
# 1325
{ return __builtin_erfcf(__x); } 
# 1328
constexpr long double erfc(long double __x) 
# 1329
{ return __builtin_erfcl(__x); } 
# 1333
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1336
erfc(_Tp __x) 
# 1337
{ return __builtin_erfc(__x); } 
# 1342
constexpr float exp2(float __x) 
# 1343
{ return __builtin_exp2f(__x); } 
# 1346
constexpr long double exp2(long double __x) 
# 1347
{ return __builtin_exp2l(__x); } 
# 1351
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1354
exp2(_Tp __x) 
# 1355
{ return __builtin_exp2(__x); } 
# 1360
constexpr float expm1(float __x) 
# 1361
{ return __builtin_expm1f(__x); } 
# 1364
constexpr long double expm1(long double __x) 
# 1365
{ return __builtin_expm1l(__x); } 
# 1369
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1372
expm1(_Tp __x) 
# 1373
{ return __builtin_expm1(__x); } 
# 1378
constexpr float fdim(float __x, float __y) 
# 1379
{ return __builtin_fdimf(__x, __y); } 
# 1382
constexpr long double fdim(long double __x, long double __y) 
# 1383
{ return __builtin_fdiml(__x, __y); } 
# 1387
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1389
fdim(_Tp __x, _Up __y) 
# 1390
{ 
# 1391
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1392
return fdim((__type)__x, (__type)__y); 
# 1393
} 
# 1398
constexpr float fma(float __x, float __y, float __z) 
# 1399
{ return __builtin_fmaf(__x, __y, __z); } 
# 1402
constexpr long double fma(long double __x, long double __y, long double __z) 
# 1403
{ return __builtin_fmal(__x, __y, __z); } 
# 1407
template< class _Tp, class _Up, class _Vp> constexpr typename __gnu_cxx::__promote_3< _Tp, _Up, _Vp> ::__type 
# 1409
fma(_Tp __x, _Up __y, _Vp __z) 
# 1410
{ 
# 1411
typedef typename __gnu_cxx::__promote_3< _Tp, _Up, _Vp> ::__type __type; 
# 1412
return fma((__type)__x, (__type)__y, (__type)__z); 
# 1413
} 
# 1418
constexpr float fmax(float __x, float __y) 
# 1419
{ return __builtin_fmaxf(__x, __y); } 
# 1422
constexpr long double fmax(long double __x, long double __y) 
# 1423
{ return __builtin_fmaxl(__x, __y); } 
# 1427
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1429
fmax(_Tp __x, _Up __y) 
# 1430
{ 
# 1431
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1432
return fmax((__type)__x, (__type)__y); 
# 1433
} 
# 1438
constexpr float fmin(float __x, float __y) 
# 1439
{ return __builtin_fminf(__x, __y); } 
# 1442
constexpr long double fmin(long double __x, long double __y) 
# 1443
{ return __builtin_fminl(__x, __y); } 
# 1447
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1449
fmin(_Tp __x, _Up __y) 
# 1450
{ 
# 1451
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1452
return fmin((__type)__x, (__type)__y); 
# 1453
} 
# 1458
constexpr float hypot(float __x, float __y) 
# 1459
{ return __builtin_hypotf(__x, __y); } 
# 1462
constexpr long double hypot(long double __x, long double __y) 
# 1463
{ return __builtin_hypotl(__x, __y); } 
# 1467
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1469
hypot(_Tp __x, _Up __y) 
# 1470
{ 
# 1471
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1472
return hypot((__type)__x, (__type)__y); 
# 1473
} 
# 1478
constexpr int ilogb(float __x) 
# 1479
{ return __builtin_ilogbf(__x); } 
# 1482
constexpr int ilogb(long double __x) 
# 1483
{ return __builtin_ilogbl(__x); } 
# 1487
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, int> ::__type 
# 1491
ilogb(_Tp __x) 
# 1492
{ return __builtin_ilogb(__x); } 
# 1497
constexpr float lgamma(float __x) 
# 1498
{ return __builtin_lgammaf(__x); } 
# 1501
constexpr long double lgamma(long double __x) 
# 1502
{ return __builtin_lgammal(__x); } 
# 1506
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1509
lgamma(_Tp __x) 
# 1510
{ return __builtin_lgamma(__x); } 
# 1515
constexpr long long llrint(float __x) 
# 1516
{ return __builtin_llrintf(__x); } 
# 1519
constexpr long long llrint(long double __x) 
# 1520
{ return __builtin_llrintl(__x); } 
# 1524
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long long> ::__type 
# 1527
llrint(_Tp __x) 
# 1528
{ return __builtin_llrint(__x); } 
# 1533
constexpr long long llround(float __x) 
# 1534
{ return __builtin_llroundf(__x); } 
# 1537
constexpr long long llround(long double __x) 
# 1538
{ return __builtin_llroundl(__x); } 
# 1542
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long long> ::__type 
# 1545
llround(_Tp __x) 
# 1546
{ return __builtin_llround(__x); } 
# 1551
constexpr float log1p(float __x) 
# 1552
{ return __builtin_log1pf(__x); } 
# 1555
constexpr long double log1p(long double __x) 
# 1556
{ return __builtin_log1pl(__x); } 
# 1560
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1563
log1p(_Tp __x) 
# 1564
{ return __builtin_log1p(__x); } 
# 1570
constexpr float log2(float __x) 
# 1571
{ return __builtin_log2f(__x); } 
# 1574
constexpr long double log2(long double __x) 
# 1575
{ return __builtin_log2l(__x); } 
# 1579
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1582
log2(_Tp __x) 
# 1583
{ return __builtin_log2(__x); } 
# 1588
constexpr float logb(float __x) 
# 1589
{ return __builtin_logbf(__x); } 
# 1592
constexpr long double logb(long double __x) 
# 1593
{ return __builtin_logbl(__x); } 
# 1597
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1600
logb(_Tp __x) 
# 1601
{ return __builtin_logb(__x); } 
# 1606
constexpr long lrint(float __x) 
# 1607
{ return __builtin_lrintf(__x); } 
# 1610
constexpr long lrint(long double __x) 
# 1611
{ return __builtin_lrintl(__x); } 
# 1615
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long> ::__type 
# 1618
lrint(_Tp __x) 
# 1619
{ return __builtin_lrint(__x); } 
# 1624
constexpr long lround(float __x) 
# 1625
{ return __builtin_lroundf(__x); } 
# 1628
constexpr long lround(long double __x) 
# 1629
{ return __builtin_lroundl(__x); } 
# 1633
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long> ::__type 
# 1636
lround(_Tp __x) 
# 1637
{ return __builtin_lround(__x); } 
# 1642
constexpr float nearbyint(float __x) 
# 1643
{ return __builtin_nearbyintf(__x); } 
# 1646
constexpr long double nearbyint(long double __x) 
# 1647
{ return __builtin_nearbyintl(__x); } 
# 1651
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1654
nearbyint(_Tp __x) 
# 1655
{ return __builtin_nearbyint(__x); } 
# 1660
constexpr float nextafter(float __x, float __y) 
# 1661
{ return __builtin_nextafterf(__x, __y); } 
# 1664
constexpr long double nextafter(long double __x, long double __y) 
# 1665
{ return __builtin_nextafterl(__x, __y); } 
# 1669
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1671
nextafter(_Tp __x, _Up __y) 
# 1672
{ 
# 1673
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1674
return nextafter((__type)__x, (__type)__y); 
# 1675
} 
# 1680
constexpr float nexttoward(float __x, long double __y) 
# 1681
{ return __builtin_nexttowardf(__x, __y); } 
# 1684
constexpr long double nexttoward(long double __x, long double __y) 
# 1685
{ return __builtin_nexttowardl(__x, __y); } 
# 1689
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1692
nexttoward(_Tp __x, long double __y) 
# 1693
{ return __builtin_nexttoward(__x, __y); } 
# 1698
constexpr float remainder(float __x, float __y) 
# 1699
{ return __builtin_remainderf(__x, __y); } 
# 1702
constexpr long double remainder(long double __x, long double __y) 
# 1703
{ return __builtin_remainderl(__x, __y); } 
# 1707
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1709
remainder(_Tp __x, _Up __y) 
# 1710
{ 
# 1711
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1712
return remainder((__type)__x, (__type)__y); 
# 1713
} 
# 1718
inline float remquo(float __x, float __y, int *__pquo) 
# 1719
{ return __builtin_remquof(__x, __y, __pquo); } 
# 1722
inline long double remquo(long double __x, long double __y, int *__pquo) 
# 1723
{ return __builtin_remquol(__x, __y, __pquo); } 
# 1727
template< class _Tp, class _Up> inline typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1729
remquo(_Tp __x, _Up __y, int *__pquo) 
# 1730
{ 
# 1731
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1732
return remquo((__type)__x, (__type)__y, __pquo); 
# 1733
} 
# 1738
constexpr float rint(float __x) 
# 1739
{ return __builtin_rintf(__x); } 
# 1742
constexpr long double rint(long double __x) 
# 1743
{ return __builtin_rintl(__x); } 
# 1747
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1750
rint(_Tp __x) 
# 1751
{ return __builtin_rint(__x); } 
# 1756
constexpr float round(float __x) 
# 1757
{ return __builtin_roundf(__x); } 
# 1760
constexpr long double round(long double __x) 
# 1761
{ return __builtin_roundl(__x); } 
# 1765
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1768
round(_Tp __x) 
# 1769
{ return __builtin_round(__x); } 
# 1774
constexpr float scalbln(float __x, long __ex) 
# 1775
{ return __builtin_scalblnf(__x, __ex); } 
# 1778
constexpr long double scalbln(long double __x, long __ex) 
# 1779
{ return __builtin_scalblnl(__x, __ex); } 
# 1783
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1786
scalbln(_Tp __x, long __ex) 
# 1787
{ return __builtin_scalbln(__x, __ex); } 
# 1792
constexpr float scalbn(float __x, int __ex) 
# 1793
{ return __builtin_scalbnf(__x, __ex); } 
# 1796
constexpr long double scalbn(long double __x, int __ex) 
# 1797
{ return __builtin_scalbnl(__x, __ex); } 
# 1801
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1804
scalbn(_Tp __x, int __ex) 
# 1805
{ return __builtin_scalbn(__x, __ex); } 
# 1810
constexpr float tgamma(float __x) 
# 1811
{ return __builtin_tgammaf(__x); } 
# 1814
constexpr long double tgamma(long double __x) 
# 1815
{ return __builtin_tgammal(__x); } 
# 1819
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1822
tgamma(_Tp __x) 
# 1823
{ return __builtin_tgamma(__x); } 
# 1828
constexpr float trunc(float __x) 
# 1829
{ return __builtin_truncf(__x); } 
# 1832
constexpr long double trunc(long double __x) 
# 1833
{ return __builtin_truncl(__x); } 
# 1837
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1840
trunc(_Tp __x) 
# 1841
{ return __builtin_trunc(__x); } 
# 1889 "/usr/include/c++/8/cmath" 3
}
# 1895
}
# 38 "/usr/include/c++/8/math.h" 3
using std::abs;
# 39
using std::acos;
# 40
using std::asin;
# 41
using std::atan;
# 42
using std::atan2;
# 43
using std::cos;
# 44
using std::sin;
# 45
using std::tan;
# 46
using std::cosh;
# 47
using std::sinh;
# 48
using std::tanh;
# 49
using std::exp;
# 50
using std::frexp;
# 51
using std::ldexp;
# 52
using std::log;
# 53
using std::log10;
# 54
using std::modf;
# 55
using std::pow;
# 56
using std::sqrt;
# 57
using std::ceil;
# 58
using std::fabs;
# 59
using std::floor;
# 60
using std::fmod;
# 63
using std::fpclassify;
# 64
using std::isfinite;
# 65
using std::isinf;
# 66
using std::isnan;
# 67
using std::isnormal;
# 68
using std::signbit;
# 69
using std::isgreater;
# 70
using std::isgreaterequal;
# 71
using std::isless;
# 72
using std::islessequal;
# 73
using std::islessgreater;
# 74
using std::isunordered;
# 78
using std::acosh;
# 79
using std::asinh;
# 80
using std::atanh;
# 81
using std::cbrt;
# 82
using std::copysign;
# 83
using std::erf;
# 84
using std::erfc;
# 85
using std::exp2;
# 86
using std::expm1;
# 87
using std::fdim;
# 88
using std::fma;
# 89
using std::fmax;
# 90
using std::fmin;
# 91
using std::hypot;
# 92
using std::ilogb;
# 93
using std::lgamma;
# 94
using std::llrint;
# 95
using std::llround;
# 96
using std::log1p;
# 97
using std::log2;
# 98
using std::logb;
# 99
using std::lrint;
# 100
using std::lround;
# 101
using std::nearbyint;
# 102
using std::nextafter;
# 103
using std::nexttoward;
# 104
using std::remainder;
# 105
using std::remquo;
# 106
using std::rint;
# 107
using std::round;
# 108
using std::scalbln;
# 109
using std::scalbn;
# 110
using std::tgamma;
# 111
using std::trunc;
# 121 "/usr/include/c++/8/cstdlib" 3
extern "C++" {
# 123
namespace std __attribute((__visibility__("default"))) { 
# 127
using ::div_t;
# 128
using ::ldiv_t;
# 130
using ::abort;
# 134
using ::atexit;
# 137
using ::at_quick_exit;
# 140
using ::atof;
# 141
using ::atoi;
# 142
using ::atol;
# 143
using ::bsearch;
# 144
using ::calloc;
# 145
using ::div;
# 146
using ::exit;
# 147
using ::free;
# 148
using ::getenv;
# 149
using ::labs;
# 150
using ::ldiv;
# 151
using ::malloc;
# 153
using ::mblen;
# 154
using ::mbstowcs;
# 155
using ::mbtowc;
# 157
using ::qsort;
# 160
using ::quick_exit;
# 163
using ::rand;
# 164
using ::realloc;
# 165
using ::srand;
# 166
using ::strtod;
# 167
using ::strtol;
# 168
using ::strtoul;
# 169
using ::system;
# 171
using ::wcstombs;
# 172
using ::wctomb;
# 177
inline ldiv_t div(long __i, long __j) { return ldiv(__i, __j); } 
# 182
}
# 195 "/usr/include/c++/8/cstdlib" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 200
using ::lldiv_t;
# 206
using ::_Exit;
# 210
using ::llabs;
# 213
inline lldiv_t div(long long __n, long long __d) 
# 214
{ lldiv_t __q; (__q.quot) = (__n / __d); (__q.rem) = (__n % __d); return __q; } 
# 216
using ::lldiv;
# 227 "/usr/include/c++/8/cstdlib" 3
using ::atoll;
# 228
using ::strtoll;
# 229
using ::strtoull;
# 231
using ::strtof;
# 232
using ::strtold;
# 235
}
# 237
namespace std { 
# 240
using __gnu_cxx::lldiv_t;
# 242
using __gnu_cxx::_Exit;
# 244
using __gnu_cxx::llabs;
# 245
using __gnu_cxx::div;
# 246
using __gnu_cxx::lldiv;
# 248
using __gnu_cxx::atoll;
# 249
using __gnu_cxx::strtof;
# 250
using __gnu_cxx::strtoll;
# 251
using __gnu_cxx::strtoull;
# 252
using __gnu_cxx::strtold;
# 253
}
# 257
}
# 38 "/usr/include/c++/8/stdlib.h" 3
using std::abort;
# 39
using std::atexit;
# 40
using std::exit;
# 43
using std::at_quick_exit;
# 46
using std::quick_exit;
# 54
using std::abs;
# 55
using std::atof;
# 56
using std::atoi;
# 57
using std::atol;
# 58
using std::bsearch;
# 59
using std::calloc;
# 60
using std::div;
# 61
using std::free;
# 62
using std::getenv;
# 63
using std::labs;
# 64
using std::ldiv;
# 65
using std::malloc;
# 67
using std::mblen;
# 68
using std::mbstowcs;
# 69
using std::mbtowc;
# 71
using std::qsort;
# 72
using std::rand;
# 73
using std::realloc;
# 74
using std::srand;
# 75
using std::strtod;
# 76
using std::strtol;
# 77
using std::strtoul;
# 78
using std::system;
# 80
using std::wcstombs;
# 81
using std::wctomb;
# 10622 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
namespace std { 
# 10623
constexpr bool signbit(float x); 
# 10624
constexpr bool signbit(double x); 
# 10625
constexpr bool signbit(long double x); 
# 10626
constexpr bool isfinite(float x); 
# 10627
constexpr bool isfinite(double x); 
# 10628
constexpr bool isfinite(long double x); 
# 10629
constexpr bool isnan(float x); 
# 10634
constexpr bool isnan(double x); 
# 10636
constexpr bool isnan(long double x); 
# 10637
constexpr bool isinf(float x); 
# 10642
constexpr bool isinf(double x); 
# 10644
constexpr bool isinf(long double x); 
# 10645
}
# 10798 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
namespace std { 
# 10800
template< class T> extern T __pow_helper(T, int); 
# 10801
template< class T> extern T __cmath_power(T, unsigned); 
# 10802
}
# 10804
using std::abs;
# 10805
using std::fabs;
# 10806
using std::ceil;
# 10807
using std::floor;
# 10808
using std::sqrt;
# 10810
using std::pow;
# 10812
using std::log;
# 10813
using std::log10;
# 10814
using std::fmod;
# 10815
using std::modf;
# 10816
using std::exp;
# 10817
using std::frexp;
# 10818
using std::ldexp;
# 10819
using std::asin;
# 10820
using std::sin;
# 10821
using std::sinh;
# 10822
using std::acos;
# 10823
using std::cos;
# 10824
using std::cosh;
# 10825
using std::atan;
# 10826
using std::atan2;
# 10827
using std::tan;
# 10828
using std::tanh;
# 11199 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
namespace std { 
# 11208 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern inline long long abs(long long); 
# 11218 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern inline long abs(long); 
# 11219
extern constexpr float abs(float); 
# 11220
extern constexpr double abs(double); 
# 11221
extern constexpr float fabs(float); 
# 11222
extern constexpr float ceil(float); 
# 11223
extern constexpr float floor(float); 
# 11224
extern constexpr float sqrt(float); 
# 11225
extern constexpr float pow(float, float); 
# 11230
template< class _Tp, class _Up> extern constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type pow(_Tp, _Up); 
# 11240
extern constexpr float log(float); 
# 11241
extern constexpr float log10(float); 
# 11242
extern constexpr float fmod(float, float); 
# 11243
extern inline float modf(float, float *); 
# 11244
extern constexpr float exp(float); 
# 11245
extern inline float frexp(float, int *); 
# 11246
extern constexpr float ldexp(float, int); 
# 11247
extern constexpr float asin(float); 
# 11248
extern constexpr float sin(float); 
# 11249
extern constexpr float sinh(float); 
# 11250
extern constexpr float acos(float); 
# 11251
extern constexpr float cos(float); 
# 11252
extern constexpr float cosh(float); 
# 11253
extern constexpr float atan(float); 
# 11254
extern constexpr float atan2(float, float); 
# 11255
extern constexpr float tan(float); 
# 11256
extern constexpr float tanh(float); 
# 11335 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
}
# 11441 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
namespace std { 
# 11442
constexpr float logb(float a); 
# 11443
constexpr int ilogb(float a); 
# 11444
constexpr float scalbn(float a, int b); 
# 11445
constexpr float scalbln(float a, long b); 
# 11446
constexpr float exp2(float a); 
# 11447
constexpr float expm1(float a); 
# 11448
constexpr float log2(float a); 
# 11449
constexpr float log1p(float a); 
# 11450
constexpr float acosh(float a); 
# 11451
constexpr float asinh(float a); 
# 11452
constexpr float atanh(float a); 
# 11453
constexpr float hypot(float a, float b); 
# 11454
constexpr float cbrt(float a); 
# 11455
constexpr float erf(float a); 
# 11456
constexpr float erfc(float a); 
# 11457
constexpr float lgamma(float a); 
# 11458
constexpr float tgamma(float a); 
# 11459
constexpr float copysign(float a, float b); 
# 11460
constexpr float nextafter(float a, float b); 
# 11461
constexpr float remainder(float a, float b); 
# 11462
inline float remquo(float a, float b, int * quo); 
# 11463
constexpr float round(float a); 
# 11464
constexpr long lround(float a); 
# 11465
constexpr long long llround(float a); 
# 11466
constexpr float trunc(float a); 
# 11467
constexpr float rint(float a); 
# 11468
constexpr long lrint(float a); 
# 11469
constexpr long long llrint(float a); 
# 11470
constexpr float nearbyint(float a); 
# 11471
constexpr float fdim(float a, float b); 
# 11472
constexpr float fma(float a, float b, float c); 
# 11473
constexpr float fmax(float a, float b); 
# 11474
constexpr float fmin(float a, float b); 
# 11475
}
# 11580 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
static inline float exp10(const float a); 
# 11582
static inline float rsqrt(const float a); 
# 11584
static inline float rcbrt(const float a); 
# 11586
static inline float sinpi(const float a); 
# 11588
static inline float cospi(const float a); 
# 11590
static inline void sincospi(const float a, float *const sptr, float *const cptr); 
# 11592
static inline void sincos(const float a, float *const sptr, float *const cptr); 
# 11594
static inline float j0(const float a); 
# 11596
static inline float j1(const float a); 
# 11598
static inline float jn(const int n, const float a); 
# 11600
static inline float y0(const float a); 
# 11602
static inline float y1(const float a); 
# 11604
static inline float yn(const int n, const float a); 
# 11606
__attribute__((unused)) static inline float cyl_bessel_i0(const float a); 
# 11608
__attribute__((unused)) static inline float cyl_bessel_i1(const float a); 
# 11610
static inline float erfinv(const float a); 
# 11612
static inline float erfcinv(const float a); 
# 11614
static inline float normcdfinv(const float a); 
# 11616
static inline float normcdf(const float a); 
# 11618
static inline float erfcx(const float a); 
# 11620
static inline double copysign(const double a, const float b); 
# 11622
static inline double copysign(const float a, const double b); 
# 11630
static inline unsigned min(const unsigned a, const unsigned b); 
# 11638
static inline unsigned min(const int a, const unsigned b); 
# 11646
static inline unsigned min(const unsigned a, const int b); 
# 11654
static inline long min(const long a, const long b); 
# 11662
static inline unsigned long min(const unsigned long a, const unsigned long b); 
# 11670
static inline unsigned long min(const long a, const unsigned long b); 
# 11678
static inline unsigned long min(const unsigned long a, const long b); 
# 11686
static inline long long min(const long long a, const long long b); 
# 11694
static inline unsigned long long min(const unsigned long long a, const unsigned long long b); 
# 11702
static inline unsigned long long min(const long long a, const unsigned long long b); 
# 11710
static inline unsigned long long min(const unsigned long long a, const long long b); 
# 11721 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
static inline float min(const float a, const float b); 
# 11732 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
static inline double min(const double a, const double b); 
# 11742 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
static inline double min(const float a, const double b); 
# 11752 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
static inline double min(const double a, const float b); 
# 11760
static inline unsigned max(const unsigned a, const unsigned b); 
# 11768
static inline unsigned max(const int a, const unsigned b); 
# 11776
static inline unsigned max(const unsigned a, const int b); 
# 11784
static inline long max(const long a, const long b); 
# 11792
static inline unsigned long max(const unsigned long a, const unsigned long b); 
# 11800
static inline unsigned long max(const long a, const unsigned long b); 
# 11808
static inline unsigned long max(const unsigned long a, const long b); 
# 11816
static inline long long max(const long long a, const long long b); 
# 11824
static inline unsigned long long max(const unsigned long long a, const unsigned long long b); 
# 11832
static inline unsigned long long max(const long long a, const unsigned long long b); 
# 11840
static inline unsigned long long max(const unsigned long long a, const long long b); 
# 11851 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
static inline float max(const float a, const float b); 
# 11862 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
static inline double max(const double a, const double b); 
# 11872 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
static inline double max(const float a, const double b); 
# 11882 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
static inline double max(const double a, const float b); 
# 11893 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
extern "C" {
# 11894
__attribute__((unused)) inline void *__nv_aligned_device_malloc(size_t size, size_t align) 
# 11895
{int volatile ___ = 1;(void)size;(void)align;
# 11898
::exit(___);}
#if 0
# 11895
{ 
# 11896
__attribute__((unused)) void *__nv_aligned_device_malloc_impl(size_t, size_t); 
# 11897
return __nv_aligned_device_malloc_impl(size, align); 
# 11898
} 
#endif
# 11899 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.h" 3
}
# 758 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.hpp" 3
static inline float exp10(const float a) 
# 759
{ 
# 760
return exp10f(a); 
# 761
} 
# 763
static inline float rsqrt(const float a) 
# 764
{ 
# 765
return rsqrtf(a); 
# 766
} 
# 768
static inline float rcbrt(const float a) 
# 769
{ 
# 770
return rcbrtf(a); 
# 771
} 
# 773
static inline float sinpi(const float a) 
# 774
{ 
# 775
return sinpif(a); 
# 776
} 
# 778
static inline float cospi(const float a) 
# 779
{ 
# 780
return cospif(a); 
# 781
} 
# 783
static inline void sincospi(const float a, float *const sptr, float *const cptr) 
# 784
{ 
# 785
sincospif(a, sptr, cptr); 
# 786
} 
# 788
static inline void sincos(const float a, float *const sptr, float *const cptr) 
# 789
{ 
# 790
sincosf(a, sptr, cptr); 
# 791
} 
# 793
static inline float j0(const float a) 
# 794
{ 
# 795
return j0f(a); 
# 796
} 
# 798
static inline float j1(const float a) 
# 799
{ 
# 800
return j1f(a); 
# 801
} 
# 803
static inline float jn(const int n, const float a) 
# 804
{ 
# 805
return jnf(n, a); 
# 806
} 
# 808
static inline float y0(const float a) 
# 809
{ 
# 810
return y0f(a); 
# 811
} 
# 813
static inline float y1(const float a) 
# 814
{ 
# 815
return y1f(a); 
# 816
} 
# 818
static inline float yn(const int n, const float a) 
# 819
{ 
# 820
return ynf(n, a); 
# 821
} 
# 823
__attribute__((unused)) static inline float cyl_bessel_i0(const float a) 
# 824
{int volatile ___ = 1;(void)a;
# 826
::exit(___);}
#if 0
# 824
{ 
# 825
return cyl_bessel_i0f(a); 
# 826
} 
#endif
# 828 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.hpp" 3
__attribute__((unused)) static inline float cyl_bessel_i1(const float a) 
# 829
{int volatile ___ = 1;(void)a;
# 831
::exit(___);}
#if 0
# 829
{ 
# 830
return cyl_bessel_i1f(a); 
# 831
} 
#endif
# 833 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.hpp" 3
static inline float erfinv(const float a) 
# 834
{ 
# 835
return erfinvf(a); 
# 836
} 
# 838
static inline float erfcinv(const float a) 
# 839
{ 
# 840
return erfcinvf(a); 
# 841
} 
# 843
static inline float normcdfinv(const float a) 
# 844
{ 
# 845
return normcdfinvf(a); 
# 846
} 
# 848
static inline float normcdf(const float a) 
# 849
{ 
# 850
return normcdff(a); 
# 851
} 
# 853
static inline float erfcx(const float a) 
# 854
{ 
# 855
return erfcxf(a); 
# 856
} 
# 858
static inline double copysign(const double a, const float b) 
# 859
{ 
# 860
return copysign(a, static_cast< double>(b)); 
# 861
} 
# 863
static inline double copysign(const float a, const double b) 
# 864
{ 
# 865
return copysign(static_cast< double>(a), b); 
# 866
} 
# 868
static inline unsigned min(const unsigned a, const unsigned b) 
# 869
{ 
# 870
return umin(a, b); 
# 871
} 
# 873
static inline unsigned min(const int a, const unsigned b) 
# 874
{ 
# 875
return umin(static_cast< unsigned>(a), b); 
# 876
} 
# 878
static inline unsigned min(const unsigned a, const int b) 
# 879
{ 
# 880
return umin(a, static_cast< unsigned>(b)); 
# 881
} 
# 883
static inline long min(const long a, const long b) 
# 884
{ 
# 885
long retval; 
# 891
if (sizeof(long) == sizeof(int)) { 
# 895
retval = (static_cast< long>(min(static_cast< int>(a), static_cast< int>(b)))); 
# 896
} else { 
# 897
retval = (static_cast< long>(llmin(static_cast< long long>(a), static_cast< long long>(b)))); 
# 898
}  
# 899
return retval; 
# 900
} 
# 902
static inline unsigned long min(const unsigned long a, const unsigned long b) 
# 903
{ 
# 904
unsigned long retval; 
# 908
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 912
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 913
} else { 
# 914
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 915
}  
# 916
return retval; 
# 917
} 
# 919
static inline unsigned long min(const long a, const unsigned long b) 
# 920
{ 
# 921
unsigned long retval; 
# 925
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 929
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 930
} else { 
# 931
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 932
}  
# 933
return retval; 
# 934
} 
# 936
static inline unsigned long min(const unsigned long a, const long b) 
# 937
{ 
# 938
unsigned long retval; 
# 942
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 946
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 947
} else { 
# 948
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 949
}  
# 950
return retval; 
# 951
} 
# 953
static inline long long min(const long long a, const long long b) 
# 954
{ 
# 955
return llmin(a, b); 
# 956
} 
# 958
static inline unsigned long long min(const unsigned long long a, const unsigned long long b) 
# 959
{ 
# 960
return ullmin(a, b); 
# 961
} 
# 963
static inline unsigned long long min(const long long a, const unsigned long long b) 
# 964
{ 
# 965
return ullmin(static_cast< unsigned long long>(a), b); 
# 966
} 
# 968
static inline unsigned long long min(const unsigned long long a, const long long b) 
# 969
{ 
# 970
return ullmin(a, static_cast< unsigned long long>(b)); 
# 971
} 
# 973
static inline float min(const float a, const float b) 
# 974
{ 
# 975
return fminf(a, b); 
# 976
} 
# 978
static inline double min(const double a, const double b) 
# 979
{ 
# 980
return fmin(a, b); 
# 981
} 
# 983
static inline double min(const float a, const double b) 
# 984
{ 
# 985
return fmin(static_cast< double>(a), b); 
# 986
} 
# 988
static inline double min(const double a, const float b) 
# 989
{ 
# 990
return fmin(a, static_cast< double>(b)); 
# 991
} 
# 993
static inline unsigned max(const unsigned a, const unsigned b) 
# 994
{ 
# 995
return umax(a, b); 
# 996
} 
# 998
static inline unsigned max(const int a, const unsigned b) 
# 999
{ 
# 1000
return umax(static_cast< unsigned>(a), b); 
# 1001
} 
# 1003
static inline unsigned max(const unsigned a, const int b) 
# 1004
{ 
# 1005
return umax(a, static_cast< unsigned>(b)); 
# 1006
} 
# 1008
static inline long max(const long a, const long b) 
# 1009
{ 
# 1010
long retval; 
# 1015
if (sizeof(long) == sizeof(int)) { 
# 1019
retval = (static_cast< long>(max(static_cast< int>(a), static_cast< int>(b)))); 
# 1020
} else { 
# 1021
retval = (static_cast< long>(llmax(static_cast< long long>(a), static_cast< long long>(b)))); 
# 1022
}  
# 1023
return retval; 
# 1024
} 
# 1026
static inline unsigned long max(const unsigned long a, const unsigned long b) 
# 1027
{ 
# 1028
unsigned long retval; 
# 1032
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1036
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 1037
} else { 
# 1038
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 1039
}  
# 1040
return retval; 
# 1041
} 
# 1043
static inline unsigned long max(const long a, const unsigned long b) 
# 1044
{ 
# 1045
unsigned long retval; 
# 1049
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1053
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 1054
} else { 
# 1055
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 1056
}  
# 1057
return retval; 
# 1058
} 
# 1060
static inline unsigned long max(const unsigned long a, const long b) 
# 1061
{ 
# 1062
unsigned long retval; 
# 1066
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1070
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 1071
} else { 
# 1072
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 1073
}  
# 1074
return retval; 
# 1075
} 
# 1077
static inline long long max(const long long a, const long long b) 
# 1078
{ 
# 1079
return llmax(a, b); 
# 1080
} 
# 1082
static inline unsigned long long max(const unsigned long long a, const unsigned long long b) 
# 1083
{ 
# 1084
return ullmax(a, b); 
# 1085
} 
# 1087
static inline unsigned long long max(const long long a, const unsigned long long b) 
# 1088
{ 
# 1089
return ullmax(static_cast< unsigned long long>(a), b); 
# 1090
} 
# 1092
static inline unsigned long long max(const unsigned long long a, const long long b) 
# 1093
{ 
# 1094
return ullmax(a, static_cast< unsigned long long>(b)); 
# 1095
} 
# 1097
static inline float max(const float a, const float b) 
# 1098
{ 
# 1099
return fmaxf(a, b); 
# 1100
} 
# 1102
static inline double max(const double a, const double b) 
# 1103
{ 
# 1104
return fmax(a, b); 
# 1105
} 
# 1107
static inline double max(const float a, const double b) 
# 1108
{ 
# 1109
return fmax(static_cast< double>(a), b); 
# 1110
} 
# 1112
static inline double max(const double a, const float b) 
# 1113
{ 
# 1114
return fmax(a, static_cast< double>(b)); 
# 1115
} 
# 1126 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/math_functions.hpp" 3
inline int min(const int a, const int b) 
# 1127
{ 
# 1128
return (a < b) ? a : b; 
# 1129
} 
# 1131
inline unsigned umin(const unsigned a, const unsigned b) 
# 1132
{ 
# 1133
return (a < b) ? a : b; 
# 1134
} 
# 1136
inline long long llmin(const long long a, const long long b) 
# 1137
{ 
# 1138
return (a < b) ? a : b; 
# 1139
} 
# 1141
inline unsigned long long ullmin(const unsigned long long a, const unsigned long long 
# 1142
b) 
# 1143
{ 
# 1144
return (a < b) ? a : b; 
# 1145
} 
# 1147
inline int max(const int a, const int b) 
# 1148
{ 
# 1149
return (a > b) ? a : b; 
# 1150
} 
# 1152
inline unsigned umax(const unsigned a, const unsigned b) 
# 1153
{ 
# 1154
return (a > b) ? a : b; 
# 1155
} 
# 1157
inline long long llmax(const long long a, const long long b) 
# 1158
{ 
# 1159
return (a > b) ? a : b; 
# 1160
} 
# 1162
inline unsigned long long ullmax(const unsigned long long a, const unsigned long long 
# 1163
b) 
# 1164
{ 
# 1165
return (a > b) ? a : b; 
# 1166
} 
# 74 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_surface_types.h" 3
template< class T, int dim = 1> 
# 75
struct surface : public surfaceReference { 
# 78
surface() 
# 79
{ 
# 80
(channelDesc) = cudaCreateChannelDesc< T> (); 
# 81
} 
# 83
surface(cudaChannelFormatDesc desc) 
# 84
{ 
# 85
(channelDesc) = desc; 
# 86
} 
# 88
}; 
# 90
template< int dim> 
# 91
struct surface< void, dim>  : public surfaceReference { 
# 94
surface() 
# 95
{ 
# 96
(channelDesc) = cudaCreateChannelDesc< void> (); 
# 97
} 
# 99
}; 
# 74 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_texture_types.h" 3
template< class T, int texType = 1, cudaTextureReadMode mode = cudaReadModeElementType> 
# 75
struct texture : public textureReference { 
# 78
texture(int norm = 0, cudaTextureFilterMode 
# 79
fMode = cudaFilterModePoint, cudaTextureAddressMode 
# 80
aMode = cudaAddressModeClamp) 
# 81
{ 
# 82
(normalized) = norm; 
# 83
(filterMode) = fMode; 
# 84
((addressMode)[0]) = aMode; 
# 85
((addressMode)[1]) = aMode; 
# 86
((addressMode)[2]) = aMode; 
# 87
(channelDesc) = cudaCreateChannelDesc< T> (); 
# 88
(sRGB) = 0; 
# 89
} 
# 91
texture(int norm, cudaTextureFilterMode 
# 92
fMode, cudaTextureAddressMode 
# 93
aMode, cudaChannelFormatDesc 
# 94
desc) 
# 95
{ 
# 96
(normalized) = norm; 
# 97
(filterMode) = fMode; 
# 98
((addressMode)[0]) = aMode; 
# 99
((addressMode)[1]) = aMode; 
# 100
((addressMode)[2]) = aMode; 
# 101
(channelDesc) = desc; 
# 102
(sRGB) = 0; 
# 103
} 
# 105
}; 
# 89 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.h" 3
extern "C" {
# 3207 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.h" 3
}
# 3229 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.h" 3
__attribute((deprecated("mulhi() is deprecated in favor of __mulhi() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress" " this warning)."))) __attribute__((unused)) static inline int mulhi(const int a, const int b); 
# 3231
__attribute((deprecated("mulhi() is deprecated in favor of __mulhi() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress" " this warning)."))) __attribute__((unused)) static inline unsigned mulhi(const unsigned a, const unsigned b); 
# 3233
__attribute((deprecated("mulhi() is deprecated in favor of __mulhi() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress" " this warning)."))) __attribute__((unused)) static inline unsigned mulhi(const int a, const unsigned b); 
# 3235
__attribute((deprecated("mulhi() is deprecated in favor of __mulhi() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress" " this warning)."))) __attribute__((unused)) static inline unsigned mulhi(const unsigned a, const int b); 
# 3237
__attribute((deprecated("mul64hi() is deprecated in favor of __mul64hi() and may be removed in a future release (Use -Wno-deprecated-declarations to supp" "ress this warning)."))) __attribute__((unused)) static inline long long mul64hi(const long long a, const long long b); 
# 3239
__attribute((deprecated("mul64hi() is deprecated in favor of __mul64hi() and may be removed in a future release (Use -Wno-deprecated-declarations to supp" "ress this warning)."))) __attribute__((unused)) static inline unsigned long long mul64hi(const unsigned long long a, const unsigned long long b); 
# 3241
__attribute((deprecated("mul64hi() is deprecated in favor of __mul64hi() and may be removed in a future release (Use -Wno-deprecated-declarations to supp" "ress this warning)."))) __attribute__((unused)) static inline unsigned long long mul64hi(const long long a, const unsigned long long b); 
# 3243
__attribute((deprecated("mul64hi() is deprecated in favor of __mul64hi() and may be removed in a future release (Use -Wno-deprecated-declarations to supp" "ress this warning)."))) __attribute__((unused)) static inline unsigned long long mul64hi(const unsigned long long a, const long long b); 
# 3245
__attribute((deprecated("float_as_int() is deprecated in favor of __float_as_int() and may be removed in a future release (Use -Wno-deprecated-declaratio" "ns to suppress this warning)."))) __attribute__((unused)) static inline int float_as_int(const float a); 
# 3247
__attribute((deprecated("int_as_float() is deprecated in favor of __int_as_float() and may be removed in a future release (Use -Wno-deprecated-declaratio" "ns to suppress this warning)."))) __attribute__((unused)) static inline float int_as_float(const int a); 
# 3249
__attribute((deprecated("float_as_uint() is deprecated in favor of __float_as_uint() and may be removed in a future release (Use -Wno-deprecated-declarat" "ions to suppress this warning)."))) __attribute__((unused)) static inline unsigned float_as_uint(const float a); 
# 3251
__attribute((deprecated("uint_as_float() is deprecated in favor of __uint_as_float() and may be removed in a future release (Use -Wno-deprecated-declarat" "ions to suppress this warning)."))) __attribute__((unused)) static inline float uint_as_float(const unsigned a); 
# 3253
__attribute((deprecated("saturate() is deprecated in favor of __saturatef() and may be removed in a future release (Use -Wno-deprecated-declarations to s" "uppress this warning)."))) __attribute__((unused)) static inline float saturate(const float a); 
# 3255
__attribute((deprecated("mul24() is deprecated in favor of __mul24() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress" " this warning)."))) __attribute__((unused)) static inline int mul24(const int a, const int b); 
# 3257
__attribute((deprecated("umul24() is deprecated in favor of __umul24() and may be removed in a future release (Use -Wno-deprecated-declarations to suppre" "ss this warning)."))) __attribute__((unused)) static inline unsigned umul24(const unsigned a, const unsigned b); 
# 3259
__attribute((deprecated("float2int() is deprecated in favor of __float2int_ru|_rd|_rn|_rz() and may be removed in a future release (Use -Wno-deprecated-d" "eclarations to suppress this warning)."))) __attribute__((unused)) static inline int float2int(const float a, const cudaRoundMode mode = cudaRoundZero); 
# 3261
__attribute((deprecated("float2uint() is deprecated in favor of __float2uint_ru|_rd|_rn|_rz() and may be removed in a future release (Use -Wno-deprecated" "-declarations to suppress this warning)."))) __attribute__((unused)) static inline unsigned float2uint(const float a, const cudaRoundMode mode = cudaRoundZero); 
# 3263
__attribute((deprecated("int2float() is deprecated in favor of __int2float_ru|_rd|_rn|_rz() and may be removed in a future release (Use -Wno-deprecated-d" "eclarations to suppress this warning)."))) __attribute__((unused)) static inline float int2float(const int a, const cudaRoundMode mode = cudaRoundNearest); 
# 3265
__attribute((deprecated("uint2float() is deprecated in favor of __uint2float_ru|_rd|_rn|_rz() and may be removed in a future release (Use -Wno-deprecated" "-declarations to suppress this warning)."))) __attribute__((unused)) static inline float uint2float(const unsigned a, const cudaRoundMode mode = cudaRoundNearest); 
# 90 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.hpp" 3
__attribute__((unused)) static inline int mulhi(const int a, const int b) 
# 91
{int volatile ___ = 1;(void)a;(void)b;
# 93
::exit(___);}
#if 0
# 91
{ 
# 92
return __mulhi(a, b); 
# 93
} 
#endif
# 95 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.hpp" 3
__attribute__((unused)) static inline unsigned mulhi(const unsigned a, const unsigned b) 
# 96
{int volatile ___ = 1;(void)a;(void)b;
# 98
::exit(___);}
#if 0
# 96
{ 
# 97
return __umulhi(a, b); 
# 98
} 
#endif
# 100 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.hpp" 3
__attribute__((unused)) static inline unsigned mulhi(const int a, const unsigned b) 
# 101
{int volatile ___ = 1;(void)a;(void)b;
# 103
::exit(___);}
#if 0
# 101
{ 
# 102
return __umulhi(static_cast< unsigned>(a), b); 
# 103
} 
#endif
# 105 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.hpp" 3
__attribute__((unused)) static inline unsigned mulhi(const unsigned a, const int b) 
# 106
{int volatile ___ = 1;(void)a;(void)b;
# 108
::exit(___);}
#if 0
# 106
{ 
# 107
return __umulhi(a, static_cast< unsigned>(b)); 
# 108
} 
#endif
# 110 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.hpp" 3
__attribute__((unused)) static inline long long mul64hi(const long long a, const long long b) 
# 111
{int volatile ___ = 1;(void)a;(void)b;
# 113
::exit(___);}
#if 0
# 111
{ 
# 112
return __mul64hi(a, b); 
# 113
} 
#endif
# 115 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.hpp" 3
__attribute__((unused)) static inline unsigned long long mul64hi(const unsigned long long a, const unsigned long long b) 
# 116
{int volatile ___ = 1;(void)a;(void)b;
# 118
::exit(___);}
#if 0
# 116
{ 
# 117
return __umul64hi(a, b); 
# 118
} 
#endif
# 120 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.hpp" 3
__attribute__((unused)) static inline unsigned long long mul64hi(const long long a, const unsigned long long b) 
# 121
{int volatile ___ = 1;(void)a;(void)b;
# 123
::exit(___);}
#if 0
# 121
{ 
# 122
return __umul64hi(static_cast< unsigned long long>(a), b); 
# 123
} 
#endif
# 125 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.hpp" 3
__attribute__((unused)) static inline unsigned long long mul64hi(const unsigned long long a, const long long b) 
# 126
{int volatile ___ = 1;(void)a;(void)b;
# 128
::exit(___);}
#if 0
# 126
{ 
# 127
return __umul64hi(a, static_cast< unsigned long long>(b)); 
# 128
} 
#endif
# 130 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.hpp" 3
__attribute__((unused)) static inline int float_as_int(const float a) 
# 131
{int volatile ___ = 1;(void)a;
# 133
::exit(___);}
#if 0
# 131
{ 
# 132
return __float_as_int(a); 
# 133
} 
#endif
# 135 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.hpp" 3
__attribute__((unused)) static inline float int_as_float(const int a) 
# 136
{int volatile ___ = 1;(void)a;
# 138
::exit(___);}
#if 0
# 136
{ 
# 137
return __int_as_float(a); 
# 138
} 
#endif
# 140 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.hpp" 3
__attribute__((unused)) static inline unsigned float_as_uint(const float a) 
# 141
{int volatile ___ = 1;(void)a;
# 143
::exit(___);}
#if 0
# 141
{ 
# 142
return __float_as_uint(a); 
# 143
} 
#endif
# 145 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.hpp" 3
__attribute__((unused)) static inline float uint_as_float(const unsigned a) 
# 146
{int volatile ___ = 1;(void)a;
# 148
::exit(___);}
#if 0
# 146
{ 
# 147
return __uint_as_float(a); 
# 148
} 
#endif
# 149 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.hpp" 3
__attribute__((unused)) static inline float saturate(const float a) 
# 150
{int volatile ___ = 1;(void)a;
# 152
::exit(___);}
#if 0
# 150
{ 
# 151
return __saturatef(a); 
# 152
} 
#endif
# 154 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.hpp" 3
__attribute__((unused)) static inline int mul24(const int a, const int b) 
# 155
{int volatile ___ = 1;(void)a;(void)b;
# 157
::exit(___);}
#if 0
# 155
{ 
# 156
return __mul24(a, b); 
# 157
} 
#endif
# 159 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.hpp" 3
__attribute__((unused)) static inline unsigned umul24(const unsigned a, const unsigned b) 
# 160
{int volatile ___ = 1;(void)a;(void)b;
# 162
::exit(___);}
#if 0
# 160
{ 
# 161
return __umul24(a, b); 
# 162
} 
#endif
# 164 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.hpp" 3
__attribute__((unused)) static inline int float2int(const float a, const cudaRoundMode mode) 
# 165
{int volatile ___ = 1;(void)a;(void)mode;
# 170
::exit(___);}
#if 0
# 165
{ 
# 166
return (mode == (cudaRoundNearest)) ? __float2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __float2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __float2int_rd(a) : __float2int_rz(a))); 
# 170
} 
#endif
# 172 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.hpp" 3
__attribute__((unused)) static inline unsigned float2uint(const float a, const cudaRoundMode mode) 
# 173
{int volatile ___ = 1;(void)a;(void)mode;
# 178
::exit(___);}
#if 0
# 173
{ 
# 174
return (mode == (cudaRoundNearest)) ? __float2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __float2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __float2uint_rd(a) : __float2uint_rz(a))); 
# 178
} 
#endif
# 180 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.hpp" 3
__attribute__((unused)) static inline float int2float(const int a, const cudaRoundMode mode) 
# 181
{int volatile ___ = 1;(void)a;(void)mode;
# 186
::exit(___);}
#if 0
# 181
{ 
# 182
return (mode == (cudaRoundZero)) ? __int2float_rz(a) : ((mode == (cudaRoundPosInf)) ? __int2float_ru(a) : ((mode == (cudaRoundMinInf)) ? __int2float_rd(a) : __int2float_rn(a))); 
# 186
} 
#endif
# 188 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.hpp" 3
__attribute__((unused)) static inline float uint2float(const unsigned a, const cudaRoundMode mode) 
# 189
{int volatile ___ = 1;(void)a;(void)mode;
# 194
::exit(___);}
#if 0
# 189
{ 
# 190
return (mode == (cudaRoundZero)) ? __uint2float_rz(a) : ((mode == (cudaRoundPosInf)) ? __uint2float_ru(a) : ((mode == (cudaRoundMinInf)) ? __uint2float_rd(a) : __uint2float_rn(a))); 
# 194
} 
#endif
# 106 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicAdd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 106
{ } 
#endif
# 108 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicAdd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 108
{ } 
#endif
# 110 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicSub(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 110
{ } 
#endif
# 112 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicSub(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 112
{ } 
#endif
# 114 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicExch(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 114
{ } 
#endif
# 116 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicExch(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 116
{ } 
#endif
# 118 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline float atomicExch(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 118
{ } 
#endif
# 120 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicMin(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 120
{ } 
#endif
# 122 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicMin(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 122
{ } 
#endif
# 124 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicMax(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 124
{ } 
#endif
# 126 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicMax(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 126
{ } 
#endif
# 128 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicInc(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 128
{ } 
#endif
# 130 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicDec(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 130
{ } 
#endif
# 132 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicAnd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 132
{ } 
#endif
# 134 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicAnd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 134
{ } 
#endif
# 136 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicOr(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 136
{ } 
#endif
# 138 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicOr(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 138
{ } 
#endif
# 140 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicXor(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 140
{ } 
#endif
# 142 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicXor(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 142
{ } 
#endif
# 144 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicCAS(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 144
{ } 
#endif
# 146 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicCAS(unsigned *address, unsigned compare, unsigned val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 146
{ } 
#endif
# 171 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
extern "C" {
# 180
}
# 189 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicAdd(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 189
{ } 
#endif
# 191 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicExch(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 191
{ } 
#endif
# 193 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicCAS(unsigned long long *address, unsigned long long compare, unsigned long long val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 193
{ } 
#endif
# 195 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute((deprecated("__any() is deprecated in favor of __any_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning)."))) __attribute__((unused)) static inline bool any(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
# 195
{ } 
#endif
# 197 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_atomic_functions.h" 3
__attribute((deprecated("__all() is deprecated in favor of __all_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning)."))) __attribute__((unused)) static inline bool all(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
# 197
{ } 
#endif
# 87 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_double_functions.h" 3
extern "C" {
# 1139 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_double_functions.h" 3
}
# 1147
__attribute__((unused)) static inline double fma(double a, double b, double c, cudaRoundMode mode); 
# 1149
__attribute__((unused)) static inline double dmul(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1151
__attribute__((unused)) static inline double dadd(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1153
__attribute__((unused)) static inline double dsub(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1155
__attribute__((unused)) static inline int double2int(double a, cudaRoundMode mode = cudaRoundZero); 
# 1157
__attribute__((unused)) static inline unsigned double2uint(double a, cudaRoundMode mode = cudaRoundZero); 
# 1159
__attribute__((unused)) static inline long long double2ll(double a, cudaRoundMode mode = cudaRoundZero); 
# 1161
__attribute__((unused)) static inline unsigned long long double2ull(double a, cudaRoundMode mode = cudaRoundZero); 
# 1163
__attribute__((unused)) static inline double ll2double(long long a, cudaRoundMode mode = cudaRoundNearest); 
# 1165
__attribute__((unused)) static inline double ull2double(unsigned long long a, cudaRoundMode mode = cudaRoundNearest); 
# 1167
__attribute__((unused)) static inline double int2double(int a, cudaRoundMode mode = cudaRoundNearest); 
# 1169
__attribute__((unused)) static inline double uint2double(unsigned a, cudaRoundMode mode = cudaRoundNearest); 
# 1171
__attribute__((unused)) static inline double float2double(float a, cudaRoundMode mode = cudaRoundNearest); 
# 93 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline double fma(double a, double b, double c, cudaRoundMode mode) 
# 94
{int volatile ___ = 1;(void)a;(void)b;(void)c;(void)mode;
# 99
::exit(___);}
#if 0
# 94
{ 
# 95
return (mode == (cudaRoundZero)) ? __fma_rz(a, b, c) : ((mode == (cudaRoundPosInf)) ? __fma_ru(a, b, c) : ((mode == (cudaRoundMinInf)) ? __fma_rd(a, b, c) : __fma_rn(a, b, c))); 
# 99
} 
#endif
# 101 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline double dmul(double a, double b, cudaRoundMode mode) 
# 102
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 107
::exit(___);}
#if 0
# 102
{ 
# 103
return (mode == (cudaRoundZero)) ? __dmul_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dmul_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dmul_rd(a, b) : __dmul_rn(a, b))); 
# 107
} 
#endif
# 109 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline double dadd(double a, double b, cudaRoundMode mode) 
# 110
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 115
::exit(___);}
#if 0
# 110
{ 
# 111
return (mode == (cudaRoundZero)) ? __dadd_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dadd_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dadd_rd(a, b) : __dadd_rn(a, b))); 
# 115
} 
#endif
# 117 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline double dsub(double a, double b, cudaRoundMode mode) 
# 118
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 123
::exit(___);}
#if 0
# 118
{ 
# 119
return (mode == (cudaRoundZero)) ? __dsub_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dsub_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dsub_rd(a, b) : __dsub_rn(a, b))); 
# 123
} 
#endif
# 125 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline int double2int(double a, cudaRoundMode mode) 
# 126
{int volatile ___ = 1;(void)a;(void)mode;
# 131
::exit(___);}
#if 0
# 126
{ 
# 127
return (mode == (cudaRoundNearest)) ? __double2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2int_rd(a) : __double2int_rz(a))); 
# 131
} 
#endif
# 133 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline unsigned double2uint(double a, cudaRoundMode mode) 
# 134
{int volatile ___ = 1;(void)a;(void)mode;
# 139
::exit(___);}
#if 0
# 134
{ 
# 135
return (mode == (cudaRoundNearest)) ? __double2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2uint_rd(a) : __double2uint_rz(a))); 
# 139
} 
#endif
# 141 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline long long double2ll(double a, cudaRoundMode mode) 
# 142
{int volatile ___ = 1;(void)a;(void)mode;
# 147
::exit(___);}
#if 0
# 142
{ 
# 143
return (mode == (cudaRoundNearest)) ? __double2ll_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ll_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ll_rd(a) : __double2ll_rz(a))); 
# 147
} 
#endif
# 149 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline unsigned long long double2ull(double a, cudaRoundMode mode) 
# 150
{int volatile ___ = 1;(void)a;(void)mode;
# 155
::exit(___);}
#if 0
# 150
{ 
# 151
return (mode == (cudaRoundNearest)) ? __double2ull_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ull_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ull_rd(a) : __double2ull_rz(a))); 
# 155
} 
#endif
# 157 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline double ll2double(long long a, cudaRoundMode mode) 
# 158
{int volatile ___ = 1;(void)a;(void)mode;
# 163
::exit(___);}
#if 0
# 158
{ 
# 159
return (mode == (cudaRoundZero)) ? __ll2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ll2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ll2double_rd(a) : __ll2double_rn(a))); 
# 163
} 
#endif
# 165 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline double ull2double(unsigned long long a, cudaRoundMode mode) 
# 166
{int volatile ___ = 1;(void)a;(void)mode;
# 171
::exit(___);}
#if 0
# 166
{ 
# 167
return (mode == (cudaRoundZero)) ? __ull2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ull2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ull2double_rd(a) : __ull2double_rn(a))); 
# 171
} 
#endif
# 173 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline double int2double(int a, cudaRoundMode mode) 
# 174
{int volatile ___ = 1;(void)a;(void)mode;
# 176
::exit(___);}
#if 0
# 174
{ 
# 175
return (double)a; 
# 176
} 
#endif
# 178 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline double uint2double(unsigned a, cudaRoundMode mode) 
# 179
{int volatile ___ = 1;(void)a;(void)mode;
# 181
::exit(___);}
#if 0
# 179
{ 
# 180
return (double)a; 
# 181
} 
#endif
# 183 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_double_functions.hpp" 3
__attribute__((unused)) static inline double float2double(float a, cudaRoundMode mode) 
# 184
{int volatile ___ = 1;(void)a;(void)mode;
# 186
::exit(___);}
#if 0
# 184
{ 
# 185
return (double)a; 
# 186
} 
#endif
# 89 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_20_atomic_functions.h" 3
__attribute__((unused)) static inline float atomicAdd(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 89
{ } 
#endif
# 100 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicMin(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 100
{ } 
#endif
# 102 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicMax(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 102
{ } 
#endif
# 104 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicAnd(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 104
{ } 
#endif
# 106 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicOr(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 106
{ } 
#endif
# 108 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicXor(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 108
{ } 
#endif
# 110 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicMin(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 110
{ } 
#endif
# 112 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicMax(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 112
{ } 
#endif
# 114 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicAnd(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 114
{ } 
#endif
# 116 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicOr(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 116
{ } 
#endif
# 118 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicXor(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 118
{ } 
#endif
# 303 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline double atomicAdd(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 303
{ } 
#endif
# 306 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicAdd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 306
{ } 
#endif
# 309 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicAdd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 309
{ } 
#endif
# 312 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicAdd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 312
{ } 
#endif
# 315 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicAdd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 315
{ } 
#endif
# 318 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicAdd_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 318
{ } 
#endif
# 321 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicAdd_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 321
{ } 
#endif
# 324 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline float atomicAdd_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 324
{ } 
#endif
# 327 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline float atomicAdd_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 327
{ } 
#endif
# 330 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline double atomicAdd_block(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 330
{ } 
#endif
# 333 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline double atomicAdd_system(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 333
{ } 
#endif
# 336 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicSub_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 336
{ } 
#endif
# 339 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicSub_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 339
{ } 
#endif
# 342 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicSub_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 342
{ } 
#endif
# 345 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicSub_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 345
{ } 
#endif
# 348 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicExch_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 348
{ } 
#endif
# 351 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicExch_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 351
{ } 
#endif
# 354 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicExch_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 354
{ } 
#endif
# 357 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicExch_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 357
{ } 
#endif
# 360 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicExch_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 360
{ } 
#endif
# 363 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicExch_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 363
{ } 
#endif
# 366 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline float atomicExch_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 366
{ } 
#endif
# 369 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline float atomicExch_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 369
{ } 
#endif
# 372 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicMin_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 372
{ } 
#endif
# 375 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicMin_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 375
{ } 
#endif
# 378 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicMin_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 378
{ } 
#endif
# 381 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicMin_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 381
{ } 
#endif
# 384 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicMin_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 384
{ } 
#endif
# 387 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicMin_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 387
{ } 
#endif
# 390 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicMin_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 390
{ } 
#endif
# 393 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicMin_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 393
{ } 
#endif
# 396 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicMax_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 396
{ } 
#endif
# 399 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicMax_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 399
{ } 
#endif
# 402 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicMax_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 402
{ } 
#endif
# 405 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicMax_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 405
{ } 
#endif
# 408 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicMax_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 408
{ } 
#endif
# 411 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicMax_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 411
{ } 
#endif
# 414 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicMax_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 414
{ } 
#endif
# 417 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicMax_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 417
{ } 
#endif
# 420 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicInc_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 420
{ } 
#endif
# 423 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicInc_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 423
{ } 
#endif
# 426 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicDec_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 426
{ } 
#endif
# 429 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicDec_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 429
{ } 
#endif
# 432 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicCAS_block(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 432
{ } 
#endif
# 435 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicCAS_system(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 435
{ } 
#endif
# 438 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicCAS_block(unsigned *address, unsigned compare, unsigned 
# 439
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 439
{ } 
#endif
# 442 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicCAS_system(unsigned *address, unsigned compare, unsigned 
# 443
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 443
{ } 
#endif
# 446 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicCAS_block(unsigned long long *address, unsigned long long 
# 447
compare, unsigned long long 
# 448
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 448
{ } 
#endif
# 451 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicCAS_system(unsigned long long *address, unsigned long long 
# 452
compare, unsigned long long 
# 453
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 453
{ } 
#endif
# 456 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicAnd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 456
{ } 
#endif
# 459 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicAnd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 459
{ } 
#endif
# 462 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicAnd_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 462
{ } 
#endif
# 465 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicAnd_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 465
{ } 
#endif
# 468 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicAnd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 468
{ } 
#endif
# 471 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicAnd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 471
{ } 
#endif
# 474 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicAnd_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 474
{ } 
#endif
# 477 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicAnd_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 477
{ } 
#endif
# 480 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicOr_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 480
{ } 
#endif
# 483 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicOr_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 483
{ } 
#endif
# 486 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicOr_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 486
{ } 
#endif
# 489 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicOr_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 489
{ } 
#endif
# 492 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicOr_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 492
{ } 
#endif
# 495 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicOr_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 495
{ } 
#endif
# 498 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicOr_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 498
{ } 
#endif
# 501 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicOr_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 501
{ } 
#endif
# 504 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicXor_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 504
{ } 
#endif
# 507 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline int atomicXor_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 507
{ } 
#endif
# 510 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicXor_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 510
{ } 
#endif
# 513 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline long long atomicXor_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 513
{ } 
#endif
# 516 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicXor_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 516
{ } 
#endif
# 519 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned atomicXor_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 519
{ } 
#endif
# 522 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicXor_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 522
{ } 
#endif
# 525 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_60_atomic_functions.h" 3
__attribute__((unused)) static inline unsigned long long atomicXor_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 525
{ } 
#endif
# 90 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_20_intrinsics.h" 3
extern "C" {
# 1503 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_20_intrinsics.h" 3
}
# 1510
__attribute((deprecated("__ballot() is deprecated in favor of __ballot_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to" " suppress this warning)."))) __attribute__((unused)) static inline unsigned ballot(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1510
{ } 
#endif
# 1512 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline int syncthreads_count(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1512
{ } 
#endif
# 1514 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline bool syncthreads_and(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1514
{ } 
#endif
# 1516 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline bool syncthreads_or(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1516
{ } 
#endif
# 1521 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __isGlobal(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1521
{ } 
#endif
# 1522 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __isShared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1522
{ } 
#endif
# 1523 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __isConstant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1523
{ } 
#endif
# 1524 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __isLocal(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1524
{ } 
#endif
# 1526 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __isGridConstant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1526
{ } 
#endif
# 1528 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline size_t __cvta_generic_to_global(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1528
{ } 
#endif
# 1529 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline size_t __cvta_generic_to_shared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1529
{ } 
#endif
# 1530 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline size_t __cvta_generic_to_constant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1530
{ } 
#endif
# 1531 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline size_t __cvta_generic_to_local(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1531
{ } 
#endif
# 1533 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline size_t __cvta_generic_to_grid_constant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1533
{ } 
#endif
# 1536 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline void *__cvta_global_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1536
{ } 
#endif
# 1537 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline void *__cvta_shared_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1537
{ } 
#endif
# 1538 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline void *__cvta_constant_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1538
{ } 
#endif
# 1539 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline void *__cvta_local_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1539
{ } 
#endif
# 1541 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_20_intrinsics.h" 3
__attribute__((unused)) static inline void *__cvta_grid_constant_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1541
{ } 
#endif
# 102 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __fns(unsigned mask, unsigned base, int offset) {int volatile ___ = 1;(void)mask;(void)base;(void)offset;::exit(___);}
#if 0
# 102
{ } 
#endif
# 103 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline void __barrier_sync(unsigned id) {int volatile ___ = 1;(void)id;::exit(___);}
#if 0
# 103
{ } 
#endif
# 104 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline void __barrier_sync_count(unsigned id, unsigned cnt) {int volatile ___ = 1;(void)id;(void)cnt;::exit(___);}
#if 0
# 104
{ } 
#endif
# 105 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline void __syncwarp(unsigned mask = 4294967295U) {int volatile ___ = 1;(void)mask;::exit(___);}
#if 0
# 105
{ } 
#endif
# 106 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline int __all_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline int __any_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 107
{ } 
#endif
# 108 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline int __uni_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __ballot_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __activemask() {int volatile ___ = 1;::exit(___);}
#if 0
# 110
{ } 
#endif
# 119 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline int __shfl(int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 119
{ } 
#endif
# 120 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned __shfl(unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 120
{ } 
#endif
# 121 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_up(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 121
{ } 
#endif
# 122 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_up(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 122
{ } 
#endif
# 123 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_down(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 123
{ } 
#endif
# 124 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_down(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 124
{ } 
#endif
# 125 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_xor(int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 125
{ } 
#endif
# 126 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_xor(unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 126
{ } 
#endif
# 127 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline float __shfl(float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 127
{ } 
#endif
# 128 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_up(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 128
{ } 
#endif
# 129 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_down(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 129
{ } 
#endif
# 130 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_xor(float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 130
{ } 
#endif
# 133 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline int __shfl_sync(unsigned mask, int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 133
{ } 
#endif
# 134 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __shfl_sync(unsigned mask, unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 134
{ } 
#endif
# 135 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline int __shfl_up_sync(unsigned mask, int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 135
{ } 
#endif
# 136 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __shfl_up_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 136
{ } 
#endif
# 137 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline int __shfl_down_sync(unsigned mask, int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 137
{ } 
#endif
# 138 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __shfl_down_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 138
{ } 
#endif
# 139 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline int __shfl_xor_sync(unsigned mask, int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 139
{ } 
#endif
# 140 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __shfl_xor_sync(unsigned mask, unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 140
{ } 
#endif
# 141 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline float __shfl_sync(unsigned mask, float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 141
{ } 
#endif
# 142 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline float __shfl_up_sync(unsigned mask, float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 142
{ } 
#endif
# 143 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline float __shfl_down_sync(unsigned mask, float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 143
{ } 
#endif
# 144 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline float __shfl_xor_sync(unsigned mask, float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 144
{ } 
#endif
# 148 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl(unsigned long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 148
{ } 
#endif
# 149 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline long long __shfl(long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 149
{ } 
#endif
# 150 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_up(long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 150
{ } 
#endif
# 151 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_up(unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 151
{ } 
#endif
# 152 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_down(long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 152
{ } 
#endif
# 153 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_down(unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 153
{ } 
#endif
# 154 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_xor(long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 154
{ } 
#endif
# 155 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_xor(unsigned long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 155
{ } 
#endif
# 156 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline double __shfl(double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 156
{ } 
#endif
# 157 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_up(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 157
{ } 
#endif
# 158 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_down(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 158
{ } 
#endif
# 159 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_xor(double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 159
{ } 
#endif
# 162 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline long long __shfl_sync(unsigned mask, long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 162
{ } 
#endif
# 163 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long long __shfl_sync(unsigned mask, unsigned long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 163
{ } 
#endif
# 164 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline long long __shfl_up_sync(unsigned mask, long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 164
{ } 
#endif
# 165 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long long __shfl_up_sync(unsigned mask, unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 165
{ } 
#endif
# 166 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline long long __shfl_down_sync(unsigned mask, long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 166
{ } 
#endif
# 167 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long long __shfl_down_sync(unsigned mask, unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 167
{ } 
#endif
# 168 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline long long __shfl_xor_sync(unsigned mask, long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 168
{ } 
#endif
# 169 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long long __shfl_xor_sync(unsigned mask, unsigned long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 169
{ } 
#endif
# 170 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline double __shfl_sync(unsigned mask, double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 170
{ } 
#endif
# 171 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline double __shfl_up_sync(unsigned mask, double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 171
{ } 
#endif
# 172 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline double __shfl_down_sync(unsigned mask, double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 172
{ } 
#endif
# 173 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline double __shfl_xor_sync(unsigned mask, double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 173
{ } 
#endif
# 177 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline long __shfl(long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 177
{ } 
#endif
# 178 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned long __shfl(unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 178
{ } 
#endif
# 179 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_up(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 179
{ } 
#endif
# 180 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_up(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 180
{ } 
#endif
# 181 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_down(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 181
{ } 
#endif
# 182 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_down(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 182
{ } 
#endif
# 183 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_xor(long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 183
{ } 
#endif
# 184 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_xor(unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 184
{ } 
#endif
# 187 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline long __shfl_sync(unsigned mask, long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 187
{ } 
#endif
# 188 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long __shfl_sync(unsigned mask, unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 188
{ } 
#endif
# 189 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline long __shfl_up_sync(unsigned mask, long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 189
{ } 
#endif
# 190 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long __shfl_up_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 190
{ } 
#endif
# 191 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline long __shfl_down_sync(unsigned mask, long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 191
{ } 
#endif
# 192 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long __shfl_down_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 192
{ } 
#endif
# 193 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline long __shfl_xor_sync(unsigned mask, long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 193
{ } 
#endif
# 194 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_30_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long __shfl_xor_sync(unsigned mask, unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 194
{ } 
#endif
# 87 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long __ldg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 87
{ } 
#endif
# 88 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long __ldg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 88
{ } 
#endif
# 90 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char __ldg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 90
{ } 
#endif
# 91 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline signed char __ldg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 91
{ } 
#endif
# 92 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short __ldg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 92
{ } 
#endif
# 93 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int __ldg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 93
{ } 
#endif
# 94 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long long __ldg(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char2 __ldg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char4 __ldg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 96
{ } 
#endif
# 97 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short2 __ldg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short4 __ldg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int2 __ldg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 99
{ } 
#endif
# 100 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int4 __ldg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 100
{ } 
#endif
# 101 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline longlong2 __ldg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 101
{ } 
#endif
# 103 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned char __ldg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 103
{ } 
#endif
# 104 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned short __ldg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 104
{ } 
#endif
# 105 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __ldg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 105
{ } 
#endif
# 106 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long long __ldg(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar2 __ldg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 107
{ } 
#endif
# 108 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar4 __ldg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort2 __ldg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort4 __ldg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 110
{ } 
#endif
# 111 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint2 __ldg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 111
{ } 
#endif
# 112 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint4 __ldg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 112
{ } 
#endif
# 113 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ulonglong2 __ldg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 113
{ } 
#endif
# 115 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float __ldg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 115
{ } 
#endif
# 116 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double __ldg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 116
{ } 
#endif
# 117 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float2 __ldg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 117
{ } 
#endif
# 118 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float4 __ldg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 118
{ } 
#endif
# 119 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double2 __ldg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 119
{ } 
#endif
# 123 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long __ldcg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 123
{ } 
#endif
# 124 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long __ldcg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 124
{ } 
#endif
# 126 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char __ldcg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 126
{ } 
#endif
# 127 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline signed char __ldcg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 127
{ } 
#endif
# 128 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short __ldcg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 128
{ } 
#endif
# 129 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int __ldcg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 129
{ } 
#endif
# 130 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long long __ldcg(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 130
{ } 
#endif
# 131 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char2 __ldcg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 131
{ } 
#endif
# 132 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char4 __ldcg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 132
{ } 
#endif
# 133 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short2 __ldcg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 133
{ } 
#endif
# 134 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short4 __ldcg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 134
{ } 
#endif
# 135 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int2 __ldcg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 135
{ } 
#endif
# 136 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int4 __ldcg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 136
{ } 
#endif
# 137 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline longlong2 __ldcg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 137
{ } 
#endif
# 139 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned char __ldcg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 139
{ } 
#endif
# 140 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned short __ldcg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 140
{ } 
#endif
# 141 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __ldcg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 141
{ } 
#endif
# 142 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long long __ldcg(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 142
{ } 
#endif
# 143 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar2 __ldcg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 143
{ } 
#endif
# 144 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar4 __ldcg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 144
{ } 
#endif
# 145 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort2 __ldcg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 145
{ } 
#endif
# 146 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort4 __ldcg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 146
{ } 
#endif
# 147 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint2 __ldcg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 147
{ } 
#endif
# 148 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint4 __ldcg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 148
{ } 
#endif
# 149 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ulonglong2 __ldcg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 149
{ } 
#endif
# 151 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float __ldcg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 151
{ } 
#endif
# 152 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double __ldcg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 152
{ } 
#endif
# 153 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float2 __ldcg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 153
{ } 
#endif
# 154 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float4 __ldcg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 154
{ } 
#endif
# 155 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double2 __ldcg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 155
{ } 
#endif
# 159 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long __ldca(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 159
{ } 
#endif
# 160 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long __ldca(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 160
{ } 
#endif
# 162 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char __ldca(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 162
{ } 
#endif
# 163 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline signed char __ldca(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 163
{ } 
#endif
# 164 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short __ldca(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 164
{ } 
#endif
# 165 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int __ldca(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 165
{ } 
#endif
# 166 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long long __ldca(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 166
{ } 
#endif
# 167 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char2 __ldca(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 167
{ } 
#endif
# 168 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char4 __ldca(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 168
{ } 
#endif
# 169 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short2 __ldca(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 169
{ } 
#endif
# 170 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short4 __ldca(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 170
{ } 
#endif
# 171 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int2 __ldca(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 171
{ } 
#endif
# 172 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int4 __ldca(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 172
{ } 
#endif
# 173 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline longlong2 __ldca(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 173
{ } 
#endif
# 175 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned char __ldca(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 175
{ } 
#endif
# 176 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned short __ldca(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 176
{ } 
#endif
# 177 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __ldca(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 177
{ } 
#endif
# 178 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long long __ldca(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 178
{ } 
#endif
# 179 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar2 __ldca(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 179
{ } 
#endif
# 180 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar4 __ldca(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 180
{ } 
#endif
# 181 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort2 __ldca(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 181
{ } 
#endif
# 182 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort4 __ldca(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 182
{ } 
#endif
# 183 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint2 __ldca(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 183
{ } 
#endif
# 184 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint4 __ldca(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 184
{ } 
#endif
# 185 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ulonglong2 __ldca(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 185
{ } 
#endif
# 187 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float __ldca(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 187
{ } 
#endif
# 188 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double __ldca(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 188
{ } 
#endif
# 189 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float2 __ldca(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 189
{ } 
#endif
# 190 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float4 __ldca(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 190
{ } 
#endif
# 191 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double2 __ldca(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 191
{ } 
#endif
# 195 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long __ldcs(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 195
{ } 
#endif
# 196 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long __ldcs(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 196
{ } 
#endif
# 198 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char __ldcs(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 198
{ } 
#endif
# 199 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline signed char __ldcs(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 199
{ } 
#endif
# 200 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short __ldcs(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 200
{ } 
#endif
# 201 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int __ldcs(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 201
{ } 
#endif
# 202 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long long __ldcs(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 202
{ } 
#endif
# 203 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char2 __ldcs(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 203
{ } 
#endif
# 204 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char4 __ldcs(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 204
{ } 
#endif
# 205 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short2 __ldcs(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 205
{ } 
#endif
# 206 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short4 __ldcs(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 206
{ } 
#endif
# 207 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int2 __ldcs(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 207
{ } 
#endif
# 208 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int4 __ldcs(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 208
{ } 
#endif
# 209 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline longlong2 __ldcs(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 209
{ } 
#endif
# 211 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned char __ldcs(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 211
{ } 
#endif
# 212 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned short __ldcs(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 212
{ } 
#endif
# 213 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __ldcs(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 213
{ } 
#endif
# 214 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long long __ldcs(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 214
{ } 
#endif
# 215 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar2 __ldcs(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 215
{ } 
#endif
# 216 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar4 __ldcs(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 216
{ } 
#endif
# 217 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort2 __ldcs(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 217
{ } 
#endif
# 218 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort4 __ldcs(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 218
{ } 
#endif
# 219 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint2 __ldcs(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 219
{ } 
#endif
# 220 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint4 __ldcs(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 220
{ } 
#endif
# 221 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ulonglong2 __ldcs(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 221
{ } 
#endif
# 223 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float __ldcs(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 223
{ } 
#endif
# 224 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double __ldcs(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 224
{ } 
#endif
# 225 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float2 __ldcs(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 225
{ } 
#endif
# 226 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float4 __ldcs(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 226
{ } 
#endif
# 227 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double2 __ldcs(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 227
{ } 
#endif
# 231 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long __ldlu(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 231
{ } 
#endif
# 232 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long __ldlu(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 232
{ } 
#endif
# 234 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char __ldlu(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 234
{ } 
#endif
# 235 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline signed char __ldlu(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 235
{ } 
#endif
# 236 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short __ldlu(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 236
{ } 
#endif
# 237 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int __ldlu(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 237
{ } 
#endif
# 238 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long long __ldlu(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 238
{ } 
#endif
# 239 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char2 __ldlu(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 239
{ } 
#endif
# 240 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char4 __ldlu(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 240
{ } 
#endif
# 241 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short2 __ldlu(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 241
{ } 
#endif
# 242 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short4 __ldlu(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 242
{ } 
#endif
# 243 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int2 __ldlu(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 243
{ } 
#endif
# 244 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int4 __ldlu(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 244
{ } 
#endif
# 245 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline longlong2 __ldlu(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 245
{ } 
#endif
# 247 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned char __ldlu(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 247
{ } 
#endif
# 248 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned short __ldlu(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 248
{ } 
#endif
# 249 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __ldlu(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 249
{ } 
#endif
# 250 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long long __ldlu(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 250
{ } 
#endif
# 251 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar2 __ldlu(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 251
{ } 
#endif
# 252 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar4 __ldlu(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 252
{ } 
#endif
# 253 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort2 __ldlu(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 253
{ } 
#endif
# 254 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort4 __ldlu(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 254
{ } 
#endif
# 255 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint2 __ldlu(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 255
{ } 
#endif
# 256 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint4 __ldlu(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 256
{ } 
#endif
# 257 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ulonglong2 __ldlu(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 257
{ } 
#endif
# 259 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float __ldlu(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 259
{ } 
#endif
# 260 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double __ldlu(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 260
{ } 
#endif
# 261 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float2 __ldlu(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 261
{ } 
#endif
# 262 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float4 __ldlu(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 262
{ } 
#endif
# 263 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double2 __ldlu(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 263
{ } 
#endif
# 267 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long __ldcv(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 267
{ } 
#endif
# 268 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long __ldcv(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 268
{ } 
#endif
# 270 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char __ldcv(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 270
{ } 
#endif
# 271 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline signed char __ldcv(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 271
{ } 
#endif
# 272 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short __ldcv(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 272
{ } 
#endif
# 273 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int __ldcv(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 273
{ } 
#endif
# 274 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline long long __ldcv(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 274
{ } 
#endif
# 275 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char2 __ldcv(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 275
{ } 
#endif
# 276 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline char4 __ldcv(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 276
{ } 
#endif
# 277 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short2 __ldcv(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 277
{ } 
#endif
# 278 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline short4 __ldcv(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 278
{ } 
#endif
# 279 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int2 __ldcv(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 279
{ } 
#endif
# 280 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline int4 __ldcv(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 280
{ } 
#endif
# 281 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline longlong2 __ldcv(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 281
{ } 
#endif
# 283 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned char __ldcv(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 283
{ } 
#endif
# 284 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned short __ldcv(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 284
{ } 
#endif
# 285 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __ldcv(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 285
{ } 
#endif
# 286 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned long long __ldcv(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 286
{ } 
#endif
# 287 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar2 __ldcv(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 287
{ } 
#endif
# 288 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uchar4 __ldcv(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 288
{ } 
#endif
# 289 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort2 __ldcv(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 289
{ } 
#endif
# 290 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ushort4 __ldcv(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 290
{ } 
#endif
# 291 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint2 __ldcv(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 291
{ } 
#endif
# 292 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline uint4 __ldcv(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 292
{ } 
#endif
# 293 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline ulonglong2 __ldcv(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 293
{ } 
#endif
# 295 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float __ldcv(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 295
{ } 
#endif
# 296 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double __ldcv(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 296
{ } 
#endif
# 297 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float2 __ldcv(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 297
{ } 
#endif
# 298 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline float4 __ldcv(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 298
{ } 
#endif
# 299 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline double2 __ldcv(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 299
{ } 
#endif
# 303 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 303
{ } 
#endif
# 304 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 304
{ } 
#endif
# 306 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 306
{ } 
#endif
# 307 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 307
{ } 
#endif
# 308 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 308
{ } 
#endif
# 309 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 309
{ } 
#endif
# 310 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(long long *ptr, long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 310
{ } 
#endif
# 311 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 311
{ } 
#endif
# 312 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 312
{ } 
#endif
# 313 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 313
{ } 
#endif
# 314 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 314
{ } 
#endif
# 315 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 315
{ } 
#endif
# 316 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 316
{ } 
#endif
# 317 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 317
{ } 
#endif
# 319 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 319
{ } 
#endif
# 320 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 320
{ } 
#endif
# 321 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 321
{ } 
#endif
# 322 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(unsigned long long *ptr, unsigned long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 322
{ } 
#endif
# 323 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 323
{ } 
#endif
# 324 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 324
{ } 
#endif
# 325 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 325
{ } 
#endif
# 326 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 326
{ } 
#endif
# 327 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 327
{ } 
#endif
# 328 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 328
{ } 
#endif
# 329 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 329
{ } 
#endif
# 331 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 331
{ } 
#endif
# 332 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 332
{ } 
#endif
# 333 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 333
{ } 
#endif
# 334 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 334
{ } 
#endif
# 335 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwb(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 335
{ } 
#endif
# 339 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 339
{ } 
#endif
# 340 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 340
{ } 
#endif
# 342 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 342
{ } 
#endif
# 343 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 343
{ } 
#endif
# 344 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 344
{ } 
#endif
# 345 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 345
{ } 
#endif
# 346 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(long long *ptr, long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 346
{ } 
#endif
# 347 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 347
{ } 
#endif
# 348 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 348
{ } 
#endif
# 349 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 349
{ } 
#endif
# 350 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 350
{ } 
#endif
# 351 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 351
{ } 
#endif
# 352 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 352
{ } 
#endif
# 353 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 353
{ } 
#endif
# 355 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 355
{ } 
#endif
# 356 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 356
{ } 
#endif
# 357 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 357
{ } 
#endif
# 358 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(unsigned long long *ptr, unsigned long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 358
{ } 
#endif
# 359 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 359
{ } 
#endif
# 360 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 360
{ } 
#endif
# 361 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 361
{ } 
#endif
# 362 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 362
{ } 
#endif
# 363 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 363
{ } 
#endif
# 364 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 364
{ } 
#endif
# 365 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 365
{ } 
#endif
# 367 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 367
{ } 
#endif
# 368 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 368
{ } 
#endif
# 369 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 369
{ } 
#endif
# 370 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 370
{ } 
#endif
# 371 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcg(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 371
{ } 
#endif
# 375 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 375
{ } 
#endif
# 376 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 376
{ } 
#endif
# 378 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 378
{ } 
#endif
# 379 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 379
{ } 
#endif
# 380 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 380
{ } 
#endif
# 381 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 381
{ } 
#endif
# 382 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(long long *ptr, long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 382
{ } 
#endif
# 383 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 383
{ } 
#endif
# 384 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 384
{ } 
#endif
# 385 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 385
{ } 
#endif
# 386 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 386
{ } 
#endif
# 387 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 387
{ } 
#endif
# 388 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 388
{ } 
#endif
# 389 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 389
{ } 
#endif
# 391 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 391
{ } 
#endif
# 392 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 392
{ } 
#endif
# 393 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 393
{ } 
#endif
# 394 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(unsigned long long *ptr, unsigned long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 394
{ } 
#endif
# 395 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 395
{ } 
#endif
# 396 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 396
{ } 
#endif
# 397 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 397
{ } 
#endif
# 398 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 398
{ } 
#endif
# 399 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 399
{ } 
#endif
# 400 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 400
{ } 
#endif
# 401 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 401
{ } 
#endif
# 403 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 403
{ } 
#endif
# 404 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 404
{ } 
#endif
# 405 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 405
{ } 
#endif
# 406 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 406
{ } 
#endif
# 407 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stcs(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 407
{ } 
#endif
# 411 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 411
{ } 
#endif
# 412 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 412
{ } 
#endif
# 414 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 414
{ } 
#endif
# 415 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 415
{ } 
#endif
# 416 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 416
{ } 
#endif
# 417 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 417
{ } 
#endif
# 418 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(long long *ptr, long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 418
{ } 
#endif
# 419 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 419
{ } 
#endif
# 420 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 420
{ } 
#endif
# 421 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 421
{ } 
#endif
# 422 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 422
{ } 
#endif
# 423 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 423
{ } 
#endif
# 424 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 424
{ } 
#endif
# 425 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 425
{ } 
#endif
# 427 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 427
{ } 
#endif
# 428 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 428
{ } 
#endif
# 429 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 429
{ } 
#endif
# 430 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(unsigned long long *ptr, unsigned long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 430
{ } 
#endif
# 431 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 431
{ } 
#endif
# 432 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 432
{ } 
#endif
# 433 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 433
{ } 
#endif
# 434 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 434
{ } 
#endif
# 435 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 435
{ } 
#endif
# 436 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 436
{ } 
#endif
# 437 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 437
{ } 
#endif
# 439 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 439
{ } 
#endif
# 440 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 440
{ } 
#endif
# 441 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 441
{ } 
#endif
# 442 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 442
{ } 
#endif
# 443 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline void __stwt(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 443
{ } 
#endif
# 460 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 460
{ } 
#endif
# 472 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __funnelshift_lc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 472
{ } 
#endif
# 485 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 485
{ } 
#endif
# 497 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_32_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __funnelshift_rc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 497
{ } 
#endif
# 89 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline int __dp2a_lo(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 89
{ } 
#endif
# 90 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __dp2a_lo(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 90
{ } 
#endif
# 92 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline int __dp2a_lo(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 92
{ } 
#endif
# 93 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __dp2a_lo(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 93
{ } 
#endif
# 95 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline int __dp2a_hi(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __dp2a_hi(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 96
{ } 
#endif
# 98 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline int __dp2a_hi(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __dp2a_hi(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 99
{ } 
#endif
# 106 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline int __dp4a(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __dp4a(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 107
{ } 
#endif
# 109 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline int __dp4a(char4 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/sm_61_intrinsics.h" 3
__attribute__((unused)) static inline unsigned __dp4a(uchar4 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 110
{ } 
#endif
# 93 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 93
{ } 
#endif
# 94 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 96
{ } 
#endif
# 97 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned long long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, long long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, float value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 99
{ } 
#endif
# 100 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, double value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 100
{ } 
#endif
# 102 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 102
{ } 
#endif
# 103 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, int value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 103
{ } 
#endif
# 104 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 104
{ } 
#endif
# 105 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 105
{ } 
#endif
# 106 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned long long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, long long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 107
{ } 
#endif
# 108 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, float value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, double value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 109
{ } 
#endif
# 111 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline void __nanosleep(unsigned ns) {int volatile ___ = 1;(void)ns;::exit(___);}
#if 0
# 111
{ } 
#endif
# 113 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_70_rt.h" 3
__attribute__((unused)) static inline unsigned short atomicCAS(unsigned short *address, unsigned short compare, unsigned short val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 113
{ } 
#endif
# 93 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_80_rt.h" 3
__attribute__((unused)) static inline unsigned __reduce_add_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 93
{ } 
#endif
# 94 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_80_rt.h" 3
__attribute__((unused)) static inline unsigned __reduce_min_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_80_rt.h" 3
__attribute__((unused)) static inline unsigned __reduce_max_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 95
{ } 
#endif
# 97 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_80_rt.h" 3
__attribute__((unused)) static inline int __reduce_add_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_80_rt.h" 3
__attribute__((unused)) static inline int __reduce_min_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_80_rt.h" 3
__attribute__((unused)) static inline int __reduce_max_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 99
{ } 
#endif
# 101 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_80_rt.h" 3
__attribute__((unused)) static inline unsigned __reduce_and_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 101
{ } 
#endif
# 102 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_80_rt.h" 3
__attribute__((unused)) static inline unsigned __reduce_or_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 102
{ } 
#endif
# 103 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_80_rt.h" 3
__attribute__((unused)) static inline unsigned __reduce_xor_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 103
{ } 
#endif
# 106 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_80_rt.h" 3
extern "C" {
# 107
__attribute__((unused)) inline void *__nv_associate_access_property(const void *ptr, unsigned long long 
# 108
property) {int volatile ___ = 1;(void)ptr;(void)property;
# 112
::exit(___);}
#if 0
# 108
{ 
# 109
__attribute__((unused)) extern void *__nv_associate_access_property_impl(const void *, unsigned long long); 
# 111
return __nv_associate_access_property_impl(ptr, property); 
# 112
} 
#endif
# 114 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_80_rt.h" 3
__attribute__((unused)) inline void __nv_memcpy_async_shared_global_4(void *dst, const void *
# 115
src, unsigned 
# 116
src_size) {int volatile ___ = 1;(void)dst;(void)src;(void)src_size;
# 121
::exit(___);}
#if 0
# 116
{ 
# 117
__attribute__((unused)) extern void __nv_memcpy_async_shared_global_4_impl(void *, const void *, unsigned); 
# 120
__nv_memcpy_async_shared_global_4_impl(dst, src, src_size); 
# 121
} 
#endif
# 123 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_80_rt.h" 3
__attribute__((unused)) inline void __nv_memcpy_async_shared_global_8(void *dst, const void *
# 124
src, unsigned 
# 125
src_size) {int volatile ___ = 1;(void)dst;(void)src;(void)src_size;
# 130
::exit(___);}
#if 0
# 125
{ 
# 126
__attribute__((unused)) extern void __nv_memcpy_async_shared_global_8_impl(void *, const void *, unsigned); 
# 129
__nv_memcpy_async_shared_global_8_impl(dst, src, src_size); 
# 130
} 
#endif
# 132 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_80_rt.h" 3
__attribute__((unused)) inline void __nv_memcpy_async_shared_global_16(void *dst, const void *
# 133
src, unsigned 
# 134
src_size) {int volatile ___ = 1;(void)dst;(void)src;(void)src_size;
# 139
::exit(___);}
#if 0
# 134
{ 
# 135
__attribute__((unused)) extern void __nv_memcpy_async_shared_global_16_impl(void *, const void *, unsigned); 
# 138
__nv_memcpy_async_shared_global_16_impl(dst, src, src_size); 
# 139
} 
#endif
# 141 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_80_rt.h" 3
}
# 89 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline unsigned __isCtaShared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 89
{ } 
#endif
# 90 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline unsigned __isClusterShared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 90
{ } 
#endif
# 91 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline void *__cluster_map_shared_rank(const void *ptr, unsigned target_block_rank) {int volatile ___ = 1;(void)ptr;(void)target_block_rank;::exit(___);}
#if 0
# 91
{ } 
#endif
# 92 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline unsigned __cluster_query_shared_rank(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 92
{ } 
#endif
# 93 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline uint2 __cluster_map_shared_multicast(const void *ptr, unsigned cluster_cta_mask) {int volatile ___ = 1;(void)ptr;(void)cluster_cta_mask;::exit(___);}
#if 0
# 93
{ } 
#endif
# 94 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline unsigned __clusterDimIsSpecified() {int volatile ___ = 1;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline dim3 __clusterDim() {int volatile ___ = 1;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline dim3 __clusterRelativeBlockIdx() {int volatile ___ = 1;::exit(___);}
#if 0
# 96
{ } 
#endif
# 97 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline dim3 __clusterGridDimInClusters() {int volatile ___ = 1;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline dim3 __clusterIdx() {int volatile ___ = 1;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline unsigned __clusterRelativeBlockRank() {int volatile ___ = 1;::exit(___);}
#if 0
# 99
{ } 
#endif
# 100 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline unsigned __clusterSizeInBlocks() {int volatile ___ = 1;::exit(___);}
#if 0
# 100
{ } 
#endif
# 101 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline void __cluster_barrier_arrive() {int volatile ___ = 1;::exit(___);}
#if 0
# 101
{ } 
#endif
# 102 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline void __cluster_barrier_wait() {int volatile ___ = 1;::exit(___);}
#if 0
# 102
{ } 
#endif
# 103 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/sm_90_rt.h" 3
__attribute__((unused)) static inline void __threadfence_cluster() {int volatile ___ = 1;::exit(___);}
#if 0
# 103
{ } 
#endif
# 122 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 123
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1Dread(T *res, surface< void, 1>  surf, int x, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 124
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)s;(void)mode;
# 128
::exit(___);}
#if 0
# 124
{ 
# 128
} 
#endif
# 130 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 131
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline T surf1Dread(surface< void, 1>  surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 132
{int volatile ___ = 1;(void)surf;(void)x;(void)mode;
# 138
::exit(___);}
#if 0
# 132
{ 
# 138
} 
#endif
# 140 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 141
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1Dread(T *res, surface< void, 1>  surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 142
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)mode;
# 146
::exit(___);}
#if 0
# 142
{ 
# 146
} 
#endif
# 149 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 150
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2Dread(T *res, surface< void, 2>  surf, int x, int y, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 151
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)s;(void)mode;
# 155
::exit(___);}
#if 0
# 151
{ 
# 155
} 
#endif
# 157 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 158
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline T surf2Dread(surface< void, 2>  surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 159
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)mode;
# 165
::exit(___);}
#if 0
# 159
{ 
# 165
} 
#endif
# 167 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 168
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2Dread(T *res, surface< void, 2>  surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 169
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)mode;
# 173
::exit(___);}
#if 0
# 169
{ 
# 173
} 
#endif
# 176 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 177
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf3Dread(T *res, surface< void, 3>  surf, int x, int y, int z, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 178
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)z;(void)s;(void)mode;
# 182
::exit(___);}
#if 0
# 178
{ 
# 182
} 
#endif
# 184 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 185
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline T surf3Dread(surface< void, 3>  surf, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 186
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)z;(void)mode;
# 192
::exit(___);}
#if 0
# 186
{ 
# 192
} 
#endif
# 194 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 195
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf3Dread(T *res, surface< void, 3>  surf, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 196
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)z;(void)mode;
# 200
::exit(___);}
#if 0
# 196
{ 
# 200
} 
#endif
# 204 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 205
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredread(T *res, surface< void, 241>  surf, int x, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 206
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)layer;(void)s;(void)mode;
# 210
::exit(___);}
#if 0
# 206
{ 
# 210
} 
#endif
# 212 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 213
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline T surf1DLayeredread(surface< void, 241>  surf, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 214
{int volatile ___ = 1;(void)surf;(void)x;(void)layer;(void)mode;
# 220
::exit(___);}
#if 0
# 214
{ 
# 220
} 
#endif
# 223 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 224
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredread(T *res, surface< void, 241>  surf, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 225
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)layer;(void)mode;
# 229
::exit(___);}
#if 0
# 225
{ 
# 229
} 
#endif
# 232 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 233
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredread(T *res, surface< void, 242>  surf, int x, int y, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 234
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layer;(void)s;(void)mode;
# 238
::exit(___);}
#if 0
# 234
{ 
# 238
} 
#endif
# 240 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 241
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline T surf2DLayeredread(surface< void, 242>  surf, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 242
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
# 248
::exit(___);}
#if 0
# 242
{ 
# 248
} 
#endif
# 251 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 252
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredread(T *res, surface< void, 242>  surf, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 253
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
# 257
::exit(___);}
#if 0
# 253
{ 
# 257
} 
#endif
# 260 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 261
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapread(T *res, surface< void, 12>  surf, int x, int y, int face, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 262
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)face;(void)s;(void)mode;
# 266
::exit(___);}
#if 0
# 262
{ 
# 266
} 
#endif
# 268 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 269
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline T surfCubemapread(surface< void, 12>  surf, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 270
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)face;(void)mode;
# 277
::exit(___);}
#if 0
# 270
{ 
# 277
} 
#endif
# 279 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 280
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapread(T *res, surface< void, 12>  surf, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 281
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)face;(void)mode;
# 285
::exit(___);}
#if 0
# 281
{ 
# 285
} 
#endif
# 288 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 289
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredread(T *res, surface< void, 252>  surf, int x, int y, int layerFace, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 290
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layerFace;(void)s;(void)mode;
# 294
::exit(___);}
#if 0
# 290
{ 
# 294
} 
#endif
# 296 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 297
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline T surfCubemapLayeredread(surface< void, 252>  surf, int x, int y, int layerFace, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 298
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
# 304
::exit(___);}
#if 0
# 298
{ 
# 304
} 
#endif
# 306 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 307
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredread(T *res, surface< void, 252>  surf, int x, int y, int layerFace, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 308
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
# 312
::exit(___);}
#if 0
# 308
{ 
# 312
} 
#endif
# 315 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 316
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1Dwrite(T val, surface< void, 1>  surf, int x, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 317
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)s;(void)mode;
# 321
::exit(___);}
#if 0
# 317
{ 
# 321
} 
#endif
# 323 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 324
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1Dwrite(T val, surface< void, 1>  surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 325
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)mode;
# 329
::exit(___);}
#if 0
# 325
{ 
# 329
} 
#endif
# 333 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 334
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2Dwrite(T val, surface< void, 2>  surf, int x, int y, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 335
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)s;(void)mode;
# 339
::exit(___);}
#if 0
# 335
{ 
# 339
} 
#endif
# 341 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 342
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2Dwrite(T val, surface< void, 2>  surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 343
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)mode;
# 347
::exit(___);}
#if 0
# 343
{ 
# 347
} 
#endif
# 350 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 351
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf3Dwrite(T val, surface< void, 3>  surf, int x, int y, int z, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 352
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)z;(void)s;(void)mode;
# 356
::exit(___);}
#if 0
# 352
{ 
# 356
} 
#endif
# 358 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 359
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf3Dwrite(T val, surface< void, 3>  surf, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 360
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)z;(void)mode;
# 364
::exit(___);}
#if 0
# 360
{ 
# 364
} 
#endif
# 367 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 368
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredwrite(T val, surface< void, 241>  surf, int x, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 369
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)layer;(void)s;(void)mode;
# 373
::exit(___);}
#if 0
# 369
{ 
# 373
} 
#endif
# 375 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 376
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredwrite(T val, surface< void, 241>  surf, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 377
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)layer;(void)mode;
# 381
::exit(___);}
#if 0
# 377
{ 
# 381
} 
#endif
# 384 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 385
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredwrite(T val, surface< void, 242>  surf, int x, int y, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 386
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layer;(void)s;(void)mode;
# 390
::exit(___);}
#if 0
# 386
{ 
# 390
} 
#endif
# 392 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 393
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredwrite(T val, surface< void, 242>  surf, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 394
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
# 398
::exit(___);}
#if 0
# 394
{ 
# 398
} 
#endif
# 401 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 402
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapwrite(T val, surface< void, 12>  surf, int x, int y, int face, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 403
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)face;(void)s;(void)mode;
# 407
::exit(___);}
#if 0
# 403
{ 
# 407
} 
#endif
# 409 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 410
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapwrite(T val, surface< void, 12>  surf, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 411
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)face;(void)mode;
# 415
::exit(___);}
#if 0
# 411
{ 
# 415
} 
#endif
# 419 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 420
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredwrite(T val, surface< void, 252>  surf, int x, int y, int layerFace, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 421
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layerFace;(void)s;(void)mode;
# 425
::exit(___);}
#if 0
# 421
{ 
# 425
} 
#endif
# 427 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_functions.h" 3
template< class T> 
# 428
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredwrite(T val, surface< void, 252>  surf, int x, int y, int layerFace, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 429
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
# 433
::exit(___);}
#if 0
# 429
{ 
# 433
} 
#endif
# 72 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 73
struct __nv_tex_rmet_ret { }; 
# 75
template<> struct __nv_tex_rmet_ret< char>  { typedef char type; }; 
# 76
template<> struct __nv_tex_rmet_ret< signed char>  { typedef signed char type; }; 
# 77
template<> struct __nv_tex_rmet_ret< unsigned char>  { typedef unsigned char type; }; 
# 78
template<> struct __nv_tex_rmet_ret< char1>  { typedef char1 type; }; 
# 79
template<> struct __nv_tex_rmet_ret< uchar1>  { typedef uchar1 type; }; 
# 80
template<> struct __nv_tex_rmet_ret< char2>  { typedef char2 type; }; 
# 81
template<> struct __nv_tex_rmet_ret< uchar2>  { typedef uchar2 type; }; 
# 82
template<> struct __nv_tex_rmet_ret< char4>  { typedef char4 type; }; 
# 83
template<> struct __nv_tex_rmet_ret< uchar4>  { typedef uchar4 type; }; 
# 85
template<> struct __nv_tex_rmet_ret< short>  { typedef short type; }; 
# 86
template<> struct __nv_tex_rmet_ret< unsigned short>  { typedef unsigned short type; }; 
# 87
template<> struct __nv_tex_rmet_ret< short1>  { typedef short1 type; }; 
# 88
template<> struct __nv_tex_rmet_ret< ushort1>  { typedef ushort1 type; }; 
# 89
template<> struct __nv_tex_rmet_ret< short2>  { typedef short2 type; }; 
# 90
template<> struct __nv_tex_rmet_ret< ushort2>  { typedef ushort2 type; }; 
# 91
template<> struct __nv_tex_rmet_ret< short4>  { typedef short4 type; }; 
# 92
template<> struct __nv_tex_rmet_ret< ushort4>  { typedef ushort4 type; }; 
# 94
template<> struct __nv_tex_rmet_ret< int>  { typedef int type; }; 
# 95
template<> struct __nv_tex_rmet_ret< unsigned>  { typedef unsigned type; }; 
# 96
template<> struct __nv_tex_rmet_ret< int1>  { typedef int1 type; }; 
# 97
template<> struct __nv_tex_rmet_ret< uint1>  { typedef uint1 type; }; 
# 98
template<> struct __nv_tex_rmet_ret< int2>  { typedef int2 type; }; 
# 99
template<> struct __nv_tex_rmet_ret< uint2>  { typedef uint2 type; }; 
# 100
template<> struct __nv_tex_rmet_ret< int4>  { typedef int4 type; }; 
# 101
template<> struct __nv_tex_rmet_ret< uint4>  { typedef uint4 type; }; 
# 113 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template<> struct __nv_tex_rmet_ret< float>  { typedef float type; }; 
# 114
template<> struct __nv_tex_rmet_ret< float1>  { typedef float1 type; }; 
# 115
template<> struct __nv_tex_rmet_ret< float2>  { typedef float2 type; }; 
# 116
template<> struct __nv_tex_rmet_ret< float4>  { typedef float4 type; }; 
# 119
template< class T> struct __nv_tex_rmet_cast { typedef T *type; }; 
# 131 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 132
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1Dfetch(texture< T, 1, cudaReadModeElementType>  t, int x) 
# 133
{int volatile ___ = 1;(void)t;(void)x;
# 139
::exit(___);}
#if 0
# 133
{ 
# 139
} 
#endif
# 141 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 142
struct __nv_tex_rmnf_ret { }; 
# 144
template<> struct __nv_tex_rmnf_ret< char>  { typedef float type; }; 
# 145
template<> struct __nv_tex_rmnf_ret< signed char>  { typedef float type; }; 
# 146
template<> struct __nv_tex_rmnf_ret< unsigned char>  { typedef float type; }; 
# 147
template<> struct __nv_tex_rmnf_ret< short>  { typedef float type; }; 
# 148
template<> struct __nv_tex_rmnf_ret< unsigned short>  { typedef float type; }; 
# 149
template<> struct __nv_tex_rmnf_ret< char1>  { typedef float1 type; }; 
# 150
template<> struct __nv_tex_rmnf_ret< uchar1>  { typedef float1 type; }; 
# 151
template<> struct __nv_tex_rmnf_ret< short1>  { typedef float1 type; }; 
# 152
template<> struct __nv_tex_rmnf_ret< ushort1>  { typedef float1 type; }; 
# 153
template<> struct __nv_tex_rmnf_ret< char2>  { typedef float2 type; }; 
# 154
template<> struct __nv_tex_rmnf_ret< uchar2>  { typedef float2 type; }; 
# 155
template<> struct __nv_tex_rmnf_ret< short2>  { typedef float2 type; }; 
# 156
template<> struct __nv_tex_rmnf_ret< ushort2>  { typedef float2 type; }; 
# 157
template<> struct __nv_tex_rmnf_ret< char4>  { typedef float4 type; }; 
# 158
template<> struct __nv_tex_rmnf_ret< uchar4>  { typedef float4 type; }; 
# 159
template<> struct __nv_tex_rmnf_ret< short4>  { typedef float4 type; }; 
# 160
template<> struct __nv_tex_rmnf_ret< ushort4>  { typedef float4 type; }; 
# 162
template< class T> 
# 163
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1Dfetch(texture< T, 1, cudaReadModeNormalizedFloat>  t, int x) 
# 164
{int volatile ___ = 1;(void)t;(void)x;
# 171
::exit(___);}
#if 0
# 164
{ 
# 171
} 
#endif
# 174 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 175
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1D(texture< T, 1, cudaReadModeElementType>  t, float x) 
# 176
{int volatile ___ = 1;(void)t;(void)x;
# 182
::exit(___);}
#if 0
# 176
{ 
# 182
} 
#endif
# 184 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 185
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1D(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x) 
# 186
{int volatile ___ = 1;(void)t;(void)x;
# 193
::exit(___);}
#if 0
# 186
{ 
# 193
} 
#endif
# 197 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 198
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2D(texture< T, 2, cudaReadModeElementType>  t, float x, float y) 
# 199
{int volatile ___ = 1;(void)t;(void)x;(void)y;
# 206
::exit(___);}
#if 0
# 199
{ 
# 206
} 
#endif
# 208 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 209
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2D(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y) 
# 210
{int volatile ___ = 1;(void)t;(void)x;(void)y;
# 217
::exit(___);}
#if 0
# 210
{ 
# 217
} 
#endif
# 221 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 222
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLayered(texture< T, 241, cudaReadModeElementType>  t, float x, int layer) 
# 223
{int volatile ___ = 1;(void)t;(void)x;(void)layer;
# 229
::exit(___);}
#if 0
# 223
{ 
# 229
} 
#endif
# 231 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 232
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLayered(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer) 
# 233
{int volatile ___ = 1;(void)t;(void)x;(void)layer;
# 240
::exit(___);}
#if 0
# 233
{ 
# 240
} 
#endif
# 244 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 245
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLayered(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer) 
# 246
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;
# 252
::exit(___);}
#if 0
# 246
{ 
# 252
} 
#endif
# 254 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 255
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLayered(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer) 
# 256
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;
# 263
::exit(___);}
#if 0
# 256
{ 
# 263
} 
#endif
# 266 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 267
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex3D(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z) 
# 268
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 274
::exit(___);}
#if 0
# 268
{ 
# 274
} 
#endif
# 276 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 277
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex3D(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z) 
# 278
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 285
::exit(___);}
#if 0
# 278
{ 
# 285
} 
#endif
# 288 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 289
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemap(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z) 
# 290
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 296
::exit(___);}
#if 0
# 290
{ 
# 296
} 
#endif
# 298 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 299
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemap(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z) 
# 300
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 307
::exit(___);}
#if 0
# 300
{ 
# 307
} 
#endif
# 310 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 311
struct __nv_tex2dgather_ret { }; 
# 312
template<> struct __nv_tex2dgather_ret< char>  { typedef char4 type; }; 
# 313
template<> struct __nv_tex2dgather_ret< signed char>  { typedef char4 type; }; 
# 314
template<> struct __nv_tex2dgather_ret< char1>  { typedef char4 type; }; 
# 315
template<> struct __nv_tex2dgather_ret< char2>  { typedef char4 type; }; 
# 316
template<> struct __nv_tex2dgather_ret< char3>  { typedef char4 type; }; 
# 317
template<> struct __nv_tex2dgather_ret< char4>  { typedef char4 type; }; 
# 318
template<> struct __nv_tex2dgather_ret< unsigned char>  { typedef uchar4 type; }; 
# 319
template<> struct __nv_tex2dgather_ret< uchar1>  { typedef uchar4 type; }; 
# 320
template<> struct __nv_tex2dgather_ret< uchar2>  { typedef uchar4 type; }; 
# 321
template<> struct __nv_tex2dgather_ret< uchar3>  { typedef uchar4 type; }; 
# 322
template<> struct __nv_tex2dgather_ret< uchar4>  { typedef uchar4 type; }; 
# 324
template<> struct __nv_tex2dgather_ret< short>  { typedef short4 type; }; 
# 325
template<> struct __nv_tex2dgather_ret< short1>  { typedef short4 type; }; 
# 326
template<> struct __nv_tex2dgather_ret< short2>  { typedef short4 type; }; 
# 327
template<> struct __nv_tex2dgather_ret< short3>  { typedef short4 type; }; 
# 328
template<> struct __nv_tex2dgather_ret< short4>  { typedef short4 type; }; 
# 329
template<> struct __nv_tex2dgather_ret< unsigned short>  { typedef ushort4 type; }; 
# 330
template<> struct __nv_tex2dgather_ret< ushort1>  { typedef ushort4 type; }; 
# 331
template<> struct __nv_tex2dgather_ret< ushort2>  { typedef ushort4 type; }; 
# 332
template<> struct __nv_tex2dgather_ret< ushort3>  { typedef ushort4 type; }; 
# 333
template<> struct __nv_tex2dgather_ret< ushort4>  { typedef ushort4 type; }; 
# 335
template<> struct __nv_tex2dgather_ret< int>  { typedef int4 type; }; 
# 336
template<> struct __nv_tex2dgather_ret< int1>  { typedef int4 type; }; 
# 337
template<> struct __nv_tex2dgather_ret< int2>  { typedef int4 type; }; 
# 338
template<> struct __nv_tex2dgather_ret< int3>  { typedef int4 type; }; 
# 339
template<> struct __nv_tex2dgather_ret< int4>  { typedef int4 type; }; 
# 340
template<> struct __nv_tex2dgather_ret< unsigned>  { typedef uint4 type; }; 
# 341
template<> struct __nv_tex2dgather_ret< uint1>  { typedef uint4 type; }; 
# 342
template<> struct __nv_tex2dgather_ret< uint2>  { typedef uint4 type; }; 
# 343
template<> struct __nv_tex2dgather_ret< uint3>  { typedef uint4 type; }; 
# 344
template<> struct __nv_tex2dgather_ret< uint4>  { typedef uint4 type; }; 
# 346
template<> struct __nv_tex2dgather_ret< float>  { typedef float4 type; }; 
# 347
template<> struct __nv_tex2dgather_ret< float1>  { typedef float4 type; }; 
# 348
template<> struct __nv_tex2dgather_ret< float2>  { typedef float4 type; }; 
# 349
template<> struct __nv_tex2dgather_ret< float3>  { typedef float4 type; }; 
# 350
template<> struct __nv_tex2dgather_ret< float4>  { typedef float4 type; }; 
# 352
template< class T> 
# 353
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex2dgather_ret< T> ::type tex2Dgather(texture< T, 2, cudaReadModeElementType>  t, float x, float y, int comp = 0) 
# 354
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)comp;
# 361
::exit(___);}
#if 0
# 354
{ 
# 361
} 
#endif
# 364 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> struct __nv_tex2dgather_rmnf_ret { }; 
# 365
template<> struct __nv_tex2dgather_rmnf_ret< char>  { typedef float4 type; }; 
# 366
template<> struct __nv_tex2dgather_rmnf_ret< signed char>  { typedef float4 type; }; 
# 367
template<> struct __nv_tex2dgather_rmnf_ret< unsigned char>  { typedef float4 type; }; 
# 368
template<> struct __nv_tex2dgather_rmnf_ret< char1>  { typedef float4 type; }; 
# 369
template<> struct __nv_tex2dgather_rmnf_ret< uchar1>  { typedef float4 type; }; 
# 370
template<> struct __nv_tex2dgather_rmnf_ret< char2>  { typedef float4 type; }; 
# 371
template<> struct __nv_tex2dgather_rmnf_ret< uchar2>  { typedef float4 type; }; 
# 372
template<> struct __nv_tex2dgather_rmnf_ret< char3>  { typedef float4 type; }; 
# 373
template<> struct __nv_tex2dgather_rmnf_ret< uchar3>  { typedef float4 type; }; 
# 374
template<> struct __nv_tex2dgather_rmnf_ret< char4>  { typedef float4 type; }; 
# 375
template<> struct __nv_tex2dgather_rmnf_ret< uchar4>  { typedef float4 type; }; 
# 376
template<> struct __nv_tex2dgather_rmnf_ret< signed short>  { typedef float4 type; }; 
# 377
template<> struct __nv_tex2dgather_rmnf_ret< unsigned short>  { typedef float4 type; }; 
# 378
template<> struct __nv_tex2dgather_rmnf_ret< short1>  { typedef float4 type; }; 
# 379
template<> struct __nv_tex2dgather_rmnf_ret< ushort1>  { typedef float4 type; }; 
# 380
template<> struct __nv_tex2dgather_rmnf_ret< short2>  { typedef float4 type; }; 
# 381
template<> struct __nv_tex2dgather_rmnf_ret< ushort2>  { typedef float4 type; }; 
# 382
template<> struct __nv_tex2dgather_rmnf_ret< short3>  { typedef float4 type; }; 
# 383
template<> struct __nv_tex2dgather_rmnf_ret< ushort3>  { typedef float4 type; }; 
# 384
template<> struct __nv_tex2dgather_rmnf_ret< short4>  { typedef float4 type; }; 
# 385
template<> struct __nv_tex2dgather_rmnf_ret< ushort4>  { typedef float4 type; }; 
# 387
template< class T> 
# 388
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex2dgather_rmnf_ret< T> ::type tex2Dgather(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, int comp = 0) 
# 389
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)comp;
# 396
::exit(___);}
#if 0
# 389
{ 
# 396
} 
#endif
# 400 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 401
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLod(texture< T, 1, cudaReadModeElementType>  t, float x, float level) 
# 402
{int volatile ___ = 1;(void)t;(void)x;(void)level;
# 408
::exit(___);}
#if 0
# 402
{ 
# 408
} 
#endif
# 410 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 411
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLod(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x, float level) 
# 412
{int volatile ___ = 1;(void)t;(void)x;(void)level;
# 419
::exit(___);}
#if 0
# 412
{ 
# 419
} 
#endif
# 422 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 423
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLod(texture< T, 2, cudaReadModeElementType>  t, float x, float y, float level) 
# 424
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)level;
# 430
::exit(___);}
#if 0
# 424
{ 
# 430
} 
#endif
# 432 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 433
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLod(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, float level) 
# 434
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)level;
# 441
::exit(___);}
#if 0
# 434
{ 
# 441
} 
#endif
# 444 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 445
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLayeredLod(texture< T, 241, cudaReadModeElementType>  t, float x, int layer, float level) 
# 446
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)level;
# 452
::exit(___);}
#if 0
# 446
{ 
# 452
} 
#endif
# 454 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 455
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLayeredLod(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer, float level) 
# 456
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)level;
# 463
::exit(___);}
#if 0
# 456
{ 
# 463
} 
#endif
# 466 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 467
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLayeredLod(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer, float level) 
# 468
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)level;
# 474
::exit(___);}
#if 0
# 468
{ 
# 474
} 
#endif
# 476 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 477
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLayeredLod(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer, float level) 
# 478
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)level;
# 485
::exit(___);}
#if 0
# 478
{ 
# 485
} 
#endif
# 488 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 489
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex3DLod(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z, float level) 
# 490
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 496
::exit(___);}
#if 0
# 490
{ 
# 496
} 
#endif
# 498 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 499
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex3DLod(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float level) 
# 500
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 507
::exit(___);}
#if 0
# 500
{ 
# 507
} 
#endif
# 510 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 511
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLod(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z, float level) 
# 512
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 518
::exit(___);}
#if 0
# 512
{ 
# 518
} 
#endif
# 520 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 521
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLod(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float level) 
# 522
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 529
::exit(___);}
#if 0
# 522
{ 
# 529
} 
#endif
# 533 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 534
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLayered(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer) 
# 535
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;
# 541
::exit(___);}
#if 0
# 535
{ 
# 541
} 
#endif
# 543 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 544
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLayered(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer) 
# 545
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;
# 552
::exit(___);}
#if 0
# 545
{ 
# 552
} 
#endif
# 556 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 557
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLayeredLod(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer, float level) 
# 558
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 564
::exit(___);}
#if 0
# 558
{ 
# 564
} 
#endif
# 566 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 567
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLayeredLod(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer, float level) 
# 568
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 575
::exit(___);}
#if 0
# 568
{ 
# 575
} 
#endif
# 579 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 580
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapGrad(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 581
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 587
::exit(___);}
#if 0
# 581
{ 
# 587
} 
#endif
# 589 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 590
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapGrad(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 591
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 598
::exit(___);}
#if 0
# 591
{ 
# 598
} 
#endif
# 602 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 603
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLayeredGrad(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 604
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 610
::exit(___);}
#if 0
# 604
{ 
# 610
} 
#endif
# 612 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 613
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLayeredGrad(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 614
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 621
::exit(___);}
#if 0
# 614
{ 
# 621
} 
#endif
# 625 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 626
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DGrad(texture< T, 1, cudaReadModeElementType>  t, float x, float dPdx, float dPdy) 
# 627
{int volatile ___ = 1;(void)t;(void)x;(void)dPdx;(void)dPdy;
# 633
::exit(___);}
#if 0
# 627
{ 
# 633
} 
#endif
# 635 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 636
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DGrad(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x, float dPdx, float dPdy) 
# 637
{int volatile ___ = 1;(void)t;(void)x;(void)dPdx;(void)dPdy;
# 644
::exit(___);}
#if 0
# 637
{ 
# 644
} 
#endif
# 648 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 649
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DGrad(texture< T, 2, cudaReadModeElementType>  t, float x, float y, float2 dPdx, float2 dPdy) 
# 650
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 656
::exit(___);}
#if 0
# 650
{ 
# 656
} 
#endif
# 658 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 659
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DGrad(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, float2 dPdx, float2 dPdy) 
# 660
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 667
::exit(___);}
#if 0
# 660
{ 
# 667
} 
#endif
# 670 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 671
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLayeredGrad(texture< T, 241, cudaReadModeElementType>  t, float x, int layer, float dPdx, float dPdy) 
# 672
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 678
::exit(___);}
#if 0
# 672
{ 
# 678
} 
#endif
# 680 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 681
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLayeredGrad(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer, float dPdx, float dPdy) 
# 682
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 689
::exit(___);}
#if 0
# 682
{ 
# 689
} 
#endif
# 692 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 693
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLayeredGrad(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 694
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 700
::exit(___);}
#if 0
# 694
{ 
# 700
} 
#endif
# 702 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 703
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLayeredGrad(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 704
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 711
::exit(___);}
#if 0
# 704
{ 
# 711
} 
#endif
# 714 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 715
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex3DGrad(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 716
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 722
::exit(___);}
#if 0
# 716
{ 
# 722
} 
#endif
# 724 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_fetch_functions.h" 3
template< class T> 
# 725
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex3DGrad(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 726
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 733
::exit(___);}
#if 0
# 726
{ 
# 733
} 
#endif
# 64 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> struct __nv_itex_trait { }; 
# 65
template<> struct __nv_itex_trait< char>  { typedef void type; }; 
# 66
template<> struct __nv_itex_trait< signed char>  { typedef void type; }; 
# 67
template<> struct __nv_itex_trait< char1>  { typedef void type; }; 
# 68
template<> struct __nv_itex_trait< char2>  { typedef void type; }; 
# 69
template<> struct __nv_itex_trait< char4>  { typedef void type; }; 
# 70
template<> struct __nv_itex_trait< unsigned char>  { typedef void type; }; 
# 71
template<> struct __nv_itex_trait< uchar1>  { typedef void type; }; 
# 72
template<> struct __nv_itex_trait< uchar2>  { typedef void type; }; 
# 73
template<> struct __nv_itex_trait< uchar4>  { typedef void type; }; 
# 74
template<> struct __nv_itex_trait< short>  { typedef void type; }; 
# 75
template<> struct __nv_itex_trait< short1>  { typedef void type; }; 
# 76
template<> struct __nv_itex_trait< short2>  { typedef void type; }; 
# 77
template<> struct __nv_itex_trait< short4>  { typedef void type; }; 
# 78
template<> struct __nv_itex_trait< unsigned short>  { typedef void type; }; 
# 79
template<> struct __nv_itex_trait< ushort1>  { typedef void type; }; 
# 80
template<> struct __nv_itex_trait< ushort2>  { typedef void type; }; 
# 81
template<> struct __nv_itex_trait< ushort4>  { typedef void type; }; 
# 82
template<> struct __nv_itex_trait< int>  { typedef void type; }; 
# 83
template<> struct __nv_itex_trait< int1>  { typedef void type; }; 
# 84
template<> struct __nv_itex_trait< int2>  { typedef void type; }; 
# 85
template<> struct __nv_itex_trait< int4>  { typedef void type; }; 
# 86
template<> struct __nv_itex_trait< unsigned>  { typedef void type; }; 
# 87
template<> struct __nv_itex_trait< uint1>  { typedef void type; }; 
# 88
template<> struct __nv_itex_trait< uint2>  { typedef void type; }; 
# 89
template<> struct __nv_itex_trait< uint4>  { typedef void type; }; 
# 100 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template<> struct __nv_itex_trait< float>  { typedef void type; }; 
# 101
template<> struct __nv_itex_trait< float1>  { typedef void type; }; 
# 102
template<> struct __nv_itex_trait< float2>  { typedef void type; }; 
# 103
template<> struct __nv_itex_trait< float4>  { typedef void type; }; 
# 107
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 108
tex1Dfetch(T *ptr, cudaTextureObject_t obj, int x) 
# 109
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;
# 113
::exit(___);}
#if 0
# 109
{ 
# 113
} 
#endif
# 115 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 116
tex1Dfetch(cudaTextureObject_t texObject, int x) 
# 117
{int volatile ___ = 1;(void)texObject;(void)x;
# 123
::exit(___);}
#if 0
# 117
{ 
# 123
} 
#endif
# 125 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 126
tex1D(T *ptr, cudaTextureObject_t obj, float x) 
# 127
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;
# 131
::exit(___);}
#if 0
# 127
{ 
# 131
} 
#endif
# 134 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 135
tex1D(cudaTextureObject_t texObject, float x) 
# 136
{int volatile ___ = 1;(void)texObject;(void)x;
# 142
::exit(___);}
#if 0
# 136
{ 
# 142
} 
#endif
# 145 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 146
tex2D(T *ptr, cudaTextureObject_t obj, float x, float y) 
# 147
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;
# 151
::exit(___);}
#if 0
# 147
{ 
# 151
} 
#endif
# 153 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 154
tex2D(cudaTextureObject_t texObject, float x, float y) 
# 155
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;
# 161
::exit(___);}
#if 0
# 155
{ 
# 161
} 
#endif
# 164 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 165
tex2D(T *ptr, cudaTextureObject_t obj, float x, float y, bool *
# 166
isResident) 
# 167
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)isResident;
# 173
::exit(___);}
#if 0
# 167
{ 
# 173
} 
#endif
# 175 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 176
tex2D(cudaTextureObject_t texObject, float x, float y, bool *isResident) 
# 177
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)isResident;
# 183
::exit(___);}
#if 0
# 177
{ 
# 183
} 
#endif
# 188 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 189
tex3D(T *ptr, cudaTextureObject_t obj, float x, float y, float z) 
# 190
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;
# 194
::exit(___);}
#if 0
# 190
{ 
# 194
} 
#endif
# 196 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 197
tex3D(cudaTextureObject_t texObject, float x, float y, float z) 
# 198
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;
# 204
::exit(___);}
#if 0
# 198
{ 
# 204
} 
#endif
# 207 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 208
tex3D(T *ptr, cudaTextureObject_t obj, float x, float y, float z, bool *
# 209
isResident) 
# 210
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)isResident;
# 216
::exit(___);}
#if 0
# 210
{ 
# 216
} 
#endif
# 218 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 219
tex3D(cudaTextureObject_t texObject, float x, float y, float z, bool *isResident) 
# 220
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)isResident;
# 226
::exit(___);}
#if 0
# 220
{ 
# 226
} 
#endif
# 230 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 231
tex1DLayered(T *ptr, cudaTextureObject_t obj, float x, int layer) 
# 232
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;
# 236
::exit(___);}
#if 0
# 232
{ 
# 236
} 
#endif
# 238 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 239
tex1DLayered(cudaTextureObject_t texObject, float x, int layer) 
# 240
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;
# 246
::exit(___);}
#if 0
# 240
{ 
# 246
} 
#endif
# 248 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 249
tex2DLayered(T *ptr, cudaTextureObject_t obj, float x, float y, int layer) 
# 250
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;
# 254
::exit(___);}
#if 0
# 250
{ 
# 254
} 
#endif
# 256 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 257
tex2DLayered(cudaTextureObject_t texObject, float x, float y, int layer) 
# 258
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;
# 264
::exit(___);}
#if 0
# 258
{ 
# 264
} 
#endif
# 267 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 268
tex2DLayered(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, bool *isResident) 
# 269
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)isResident;
# 275
::exit(___);}
#if 0
# 269
{ 
# 275
} 
#endif
# 277 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 278
tex2DLayered(cudaTextureObject_t texObject, float x, float y, int layer, bool *isResident) 
# 279
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)isResident;
# 285
::exit(___);}
#if 0
# 279
{ 
# 285
} 
#endif
# 289 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 290
texCubemap(T *ptr, cudaTextureObject_t obj, float x, float y, float z) 
# 291
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;
# 295
::exit(___);}
#if 0
# 291
{ 
# 295
} 
#endif
# 298 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 299
texCubemap(cudaTextureObject_t texObject, float x, float y, float z) 
# 300
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;
# 306
::exit(___);}
#if 0
# 300
{ 
# 306
} 
#endif
# 309 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 310
texCubemapLayered(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer) 
# 311
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;
# 315
::exit(___);}
#if 0
# 311
{ 
# 315
} 
#endif
# 317 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 318
texCubemapLayered(cudaTextureObject_t texObject, float x, float y, float z, int layer) 
# 319
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;
# 325
::exit(___);}
#if 0
# 319
{ 
# 325
} 
#endif
# 327 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 328
tex2Dgather(T *ptr, cudaTextureObject_t obj, float x, float y, int comp = 0) 
# 329
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)comp;
# 333
::exit(___);}
#if 0
# 329
{ 
# 333
} 
#endif
# 335 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 336
tex2Dgather(cudaTextureObject_t to, float x, float y, int comp = 0) 
# 337
{int volatile ___ = 1;(void)to;(void)x;(void)y;(void)comp;
# 343
::exit(___);}
#if 0
# 337
{ 
# 343
} 
#endif
# 346 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 347
tex2Dgather(T *ptr, cudaTextureObject_t obj, float x, float y, bool *isResident, int comp = 0) 
# 348
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)isResident;(void)comp;
# 354
::exit(___);}
#if 0
# 348
{ 
# 354
} 
#endif
# 356 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 357
tex2Dgather(cudaTextureObject_t to, float x, float y, bool *isResident, int comp = 0) 
# 358
{int volatile ___ = 1;(void)to;(void)x;(void)y;(void)isResident;(void)comp;
# 364
::exit(___);}
#if 0
# 358
{ 
# 364
} 
#endif
# 368 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 369
tex1DLod(T *ptr, cudaTextureObject_t obj, float x, float level) 
# 370
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)level;
# 374
::exit(___);}
#if 0
# 370
{ 
# 374
} 
#endif
# 376 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 377
tex1DLod(cudaTextureObject_t texObject, float x, float level) 
# 378
{int volatile ___ = 1;(void)texObject;(void)x;(void)level;
# 384
::exit(___);}
#if 0
# 378
{ 
# 384
} 
#endif
# 387 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 388
tex2DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float level) 
# 389
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)level;
# 393
::exit(___);}
#if 0
# 389
{ 
# 393
} 
#endif
# 395 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 396
tex2DLod(cudaTextureObject_t texObject, float x, float y, float level) 
# 397
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)level;
# 403
::exit(___);}
#if 0
# 397
{ 
# 403
} 
#endif
# 407 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 408
tex2DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float level, bool *isResident) 
# 409
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)level;(void)isResident;
# 415
::exit(___);}
#if 0
# 409
{ 
# 415
} 
#endif
# 417 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 418
tex2DLod(cudaTextureObject_t texObject, float x, float y, float level, bool *isResident) 
# 419
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)level;(void)isResident;
# 425
::exit(___);}
#if 0
# 419
{ 
# 425
} 
#endif
# 430 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 431
tex3DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level) 
# 432
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;
# 436
::exit(___);}
#if 0
# 432
{ 
# 436
} 
#endif
# 438 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 439
tex3DLod(cudaTextureObject_t texObject, float x, float y, float z, float level) 
# 440
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;
# 446
::exit(___);}
#if 0
# 440
{ 
# 446
} 
#endif
# 449 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 450
tex3DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level, bool *isResident) 
# 451
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;(void)isResident;
# 457
::exit(___);}
#if 0
# 451
{ 
# 457
} 
#endif
# 459 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 460
tex3DLod(cudaTextureObject_t texObject, float x, float y, float z, float level, bool *isResident) 
# 461
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;(void)isResident;
# 467
::exit(___);}
#if 0
# 461
{ 
# 467
} 
#endif
# 472 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 473
tex1DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, int layer, float level) 
# 474
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)level;
# 478
::exit(___);}
#if 0
# 474
{ 
# 478
} 
#endif
# 480 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 481
tex1DLayeredLod(cudaTextureObject_t texObject, float x, int layer, float level) 
# 482
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)level;
# 488
::exit(___);}
#if 0
# 482
{ 
# 488
} 
#endif
# 491 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 492
tex2DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float level) 
# 493
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)level;
# 497
::exit(___);}
#if 0
# 493
{ 
# 497
} 
#endif
# 499 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 500
tex2DLayeredLod(cudaTextureObject_t texObject, float x, float y, int layer, float level) 
# 501
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)level;
# 507
::exit(___);}
#if 0
# 501
{ 
# 507
} 
#endif
# 510 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 511
tex2DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float level, bool *isResident) 
# 512
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)level;(void)isResident;
# 518
::exit(___);}
#if 0
# 512
{ 
# 518
} 
#endif
# 520 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 521
tex2DLayeredLod(cudaTextureObject_t texObject, float x, float y, int layer, float level, bool *isResident) 
# 522
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)level;(void)isResident;
# 528
::exit(___);}
#if 0
# 522
{ 
# 528
} 
#endif
# 531 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 532
texCubemapLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level) 
# 533
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;
# 537
::exit(___);}
#if 0
# 533
{ 
# 537
} 
#endif
# 539 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 540
texCubemapLod(cudaTextureObject_t texObject, float x, float y, float z, float level) 
# 541
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;
# 547
::exit(___);}
#if 0
# 541
{ 
# 547
} 
#endif
# 550 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 551
texCubemapGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 552
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 556
::exit(___);}
#if 0
# 552
{ 
# 556
} 
#endif
# 558 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 559
texCubemapGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 560
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 566
::exit(___);}
#if 0
# 560
{ 
# 566
} 
#endif
# 568 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 569
texCubemapLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer, float level) 
# 570
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 574
::exit(___);}
#if 0
# 570
{ 
# 574
} 
#endif
# 576 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 577
texCubemapLayeredLod(cudaTextureObject_t texObject, float x, float y, float z, int layer, float level) 
# 578
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 584
::exit(___);}
#if 0
# 578
{ 
# 584
} 
#endif
# 586 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 587
tex1DGrad(T *ptr, cudaTextureObject_t obj, float x, float dPdx, float dPdy) 
# 588
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)dPdx;(void)dPdy;
# 592
::exit(___);}
#if 0
# 588
{ 
# 592
} 
#endif
# 594 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 595
tex1DGrad(cudaTextureObject_t texObject, float x, float dPdx, float dPdy) 
# 596
{int volatile ___ = 1;(void)texObject;(void)x;(void)dPdx;(void)dPdy;
# 602
::exit(___);}
#if 0
# 596
{ 
# 602
} 
#endif
# 605 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 606
tex2DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float2 dPdx, float2 dPdy) 
# 607
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 612
::exit(___);}
#if 0
# 607
{ 
# 612
} 
#endif
# 614 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 615
tex2DGrad(cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy) 
# 616
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 622
::exit(___);}
#if 0
# 616
{ 
# 622
} 
#endif
# 625 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 626
tex2DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float2 dPdx, float2 dPdy, bool *isResident) 
# 627
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)dPdx;(void)dPdy;(void)isResident;
# 634
::exit(___);}
#if 0
# 627
{ 
# 634
} 
#endif
# 636 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 637
tex2DGrad(cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy, bool *isResident) 
# 638
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)dPdx;(void)dPdy;(void)isResident;
# 644
::exit(___);}
#if 0
# 638
{ 
# 644
} 
#endif
# 648 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 649
tex3DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 650
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 654
::exit(___);}
#if 0
# 650
{ 
# 654
} 
#endif
# 656 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 657
tex3DGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 658
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 664
::exit(___);}
#if 0
# 658
{ 
# 664
} 
#endif
# 667 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 668
tex3DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy, bool *isResident) 
# 669
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;(void)isResident;
# 675
::exit(___);}
#if 0
# 669
{ 
# 675
} 
#endif
# 677 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 678
tex3DGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy, bool *isResident) 
# 679
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;(void)isResident;
# 685
::exit(___);}
#if 0
# 679
{ 
# 685
} 
#endif
# 690 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 691
tex1DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, int layer, float dPdx, float dPdy) 
# 692
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 696
::exit(___);}
#if 0
# 692
{ 
# 696
} 
#endif
# 698 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 699
tex1DLayeredGrad(cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy) 
# 700
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 706
::exit(___);}
#if 0
# 700
{ 
# 706
} 
#endif
# 709 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 710
tex2DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 711
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 715
::exit(___);}
#if 0
# 711
{ 
# 715
} 
#endif
# 717 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 718
tex2DLayeredGrad(cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 719
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 725
::exit(___);}
#if 0
# 719
{ 
# 725
} 
#endif
# 728 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 729
tex2DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float2 dPdx, float2 dPdy, bool *isResident) 
# 730
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;(void)isResident;
# 736
::exit(___);}
#if 0
# 730
{ 
# 736
} 
#endif
# 738 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 739
tex2DLayeredGrad(cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy, bool *isResident) 
# 740
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;(void)isResident;
# 746
::exit(___);}
#if 0
# 740
{ 
# 746
} 
#endif
# 750 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 751
texCubemapLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 752
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 756
::exit(___);}
#if 0
# 752
{ 
# 756
} 
#endif
# 758 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/texture_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 759
texCubemapLayeredGrad(cudaTextureObject_t texObject, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 760
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 766
::exit(___);}
#if 0
# 760
{ 
# 766
} 
#endif
# 59 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_indirect_functions.h" 3
template< class T> struct __nv_isurf_trait { }; 
# 60
template<> struct __nv_isurf_trait< char>  { typedef void type; }; 
# 61
template<> struct __nv_isurf_trait< signed char>  { typedef void type; }; 
# 62
template<> struct __nv_isurf_trait< char1>  { typedef void type; }; 
# 63
template<> struct __nv_isurf_trait< unsigned char>  { typedef void type; }; 
# 64
template<> struct __nv_isurf_trait< uchar1>  { typedef void type; }; 
# 65
template<> struct __nv_isurf_trait< short>  { typedef void type; }; 
# 66
template<> struct __nv_isurf_trait< short1>  { typedef void type; }; 
# 67
template<> struct __nv_isurf_trait< unsigned short>  { typedef void type; }; 
# 68
template<> struct __nv_isurf_trait< ushort1>  { typedef void type; }; 
# 69
template<> struct __nv_isurf_trait< int>  { typedef void type; }; 
# 70
template<> struct __nv_isurf_trait< int1>  { typedef void type; }; 
# 71
template<> struct __nv_isurf_trait< unsigned>  { typedef void type; }; 
# 72
template<> struct __nv_isurf_trait< uint1>  { typedef void type; }; 
# 73
template<> struct __nv_isurf_trait< long long>  { typedef void type; }; 
# 74
template<> struct __nv_isurf_trait< longlong1>  { typedef void type; }; 
# 75
template<> struct __nv_isurf_trait< unsigned long long>  { typedef void type; }; 
# 76
template<> struct __nv_isurf_trait< ulonglong1>  { typedef void type; }; 
# 77
template<> struct __nv_isurf_trait< float>  { typedef void type; }; 
# 78
template<> struct __nv_isurf_trait< float1>  { typedef void type; }; 
# 80
template<> struct __nv_isurf_trait< char2>  { typedef void type; }; 
# 81
template<> struct __nv_isurf_trait< uchar2>  { typedef void type; }; 
# 82
template<> struct __nv_isurf_trait< short2>  { typedef void type; }; 
# 83
template<> struct __nv_isurf_trait< ushort2>  { typedef void type; }; 
# 84
template<> struct __nv_isurf_trait< int2>  { typedef void type; }; 
# 85
template<> struct __nv_isurf_trait< uint2>  { typedef void type; }; 
# 86
template<> struct __nv_isurf_trait< longlong2>  { typedef void type; }; 
# 87
template<> struct __nv_isurf_trait< ulonglong2>  { typedef void type; }; 
# 88
template<> struct __nv_isurf_trait< float2>  { typedef void type; }; 
# 90
template<> struct __nv_isurf_trait< char4>  { typedef void type; }; 
# 91
template<> struct __nv_isurf_trait< uchar4>  { typedef void type; }; 
# 92
template<> struct __nv_isurf_trait< short4>  { typedef void type; }; 
# 93
template<> struct __nv_isurf_trait< ushort4>  { typedef void type; }; 
# 94
template<> struct __nv_isurf_trait< int4>  { typedef void type; }; 
# 95
template<> struct __nv_isurf_trait< uint4>  { typedef void type; }; 
# 96
template<> struct __nv_isurf_trait< float4>  { typedef void type; }; 
# 99
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 100
surf1Dread(T *ptr, cudaSurfaceObject_t obj, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 101
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)mode;
# 105
::exit(___);}
#if 0
# 101
{ 
# 105
} 
#endif
# 107 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 108
surf1Dread(cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 109
{int volatile ___ = 1;(void)surfObject;(void)x;(void)boundaryMode;
# 115
::exit(___);}
#if 0
# 109
{ 
# 115
} 
#endif
# 117 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 118
surf2Dread(T *ptr, cudaSurfaceObject_t obj, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 119
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)mode;
# 123
::exit(___);}
#if 0
# 119
{ 
# 123
} 
#endif
# 125 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 126
surf2Dread(cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 127
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)boundaryMode;
# 133
::exit(___);}
#if 0
# 127
{ 
# 133
} 
#endif
# 136 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 137
surf3Dread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 138
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)mode;
# 142
::exit(___);}
#if 0
# 138
{ 
# 142
} 
#endif
# 144 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 145
surf3Dread(cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 146
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)z;(void)boundaryMode;
# 152
::exit(___);}
#if 0
# 146
{ 
# 152
} 
#endif
# 154 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 155
surf1DLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 156
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)mode;
# 160
::exit(___);}
#if 0
# 156
{ 
# 160
} 
#endif
# 162 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 163
surf1DLayeredread(cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 164
{int volatile ___ = 1;(void)surfObject;(void)x;(void)layer;(void)boundaryMode;
# 170
::exit(___);}
#if 0
# 164
{ 
# 170
} 
#endif
# 172 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 173
surf2DLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 174
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)mode;
# 178
::exit(___);}
#if 0
# 174
{ 
# 178
} 
#endif
# 180 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 181
surf2DLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 182
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layer;(void)boundaryMode;
# 188
::exit(___);}
#if 0
# 182
{ 
# 188
} 
#endif
# 190 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 191
surfCubemapread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 192
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)face;(void)mode;
# 196
::exit(___);}
#if 0
# 192
{ 
# 196
} 
#endif
# 198 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 199
surfCubemapread(cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 200
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)face;(void)boundaryMode;
# 206
::exit(___);}
#if 0
# 200
{ 
# 206
} 
#endif
# 208 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 209
surfCubemapLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int layerface, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 210
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;
# 214
::exit(___);}
#if 0
# 210
{ 
# 214
} 
#endif
# 216 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static T 
# 217
surfCubemapLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 218
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layerface;(void)boundaryMode;
# 224
::exit(___);}
#if 0
# 218
{ 
# 224
} 
#endif
# 226 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 227
surf1Dwrite(T val, cudaSurfaceObject_t obj, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 228
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)mode;
# 232
::exit(___);}
#if 0
# 228
{ 
# 232
} 
#endif
# 234 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 235
surf2Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 236
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)mode;
# 240
::exit(___);}
#if 0
# 236
{ 
# 240
} 
#endif
# 242 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 243
surf3Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 244
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)z;(void)mode;
# 248
::exit(___);}
#if 0
# 244
{ 
# 248
} 
#endif
# 250 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 251
surf1DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 252
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)layer;(void)mode;
# 256
::exit(___);}
#if 0
# 252
{ 
# 256
} 
#endif
# 258 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 259
surf2DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 260
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layer;(void)mode;
# 264
::exit(___);}
#if 0
# 260
{ 
# 264
} 
#endif
# 266 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 267
surfCubemapwrite(T val, cudaSurfaceObject_t obj, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 268
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)face;(void)mode;
# 272
::exit(___);}
#if 0
# 268
{ 
# 272
} 
#endif
# 274 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/surface_indirect_functions.h" 3
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 275
surfCubemapLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layerface, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 276
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;
# 280
::exit(___);}
#if 0
# 276
{ 
# 280
} 
#endif
# 3309 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/crt/device_functions.h" 3
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, CUstream_st * stream = 0); 
# 68 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/device_launch_parameters.h" 3
extern "C" {
# 71
extern const uint3 __device_builtin_variable_threadIdx; 
# 72
extern const uint3 __device_builtin_variable_blockIdx; 
# 73
extern const dim3 __device_builtin_variable_blockDim; 
# 74
extern const dim3 __device_builtin_variable_gridDim; 
# 75
extern const int __device_builtin_variable_warpSize; 
# 80
}
# 67 "/usr/include/c++/8/bits/stl_relops.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 71
namespace rel_ops { 
# 85 "/usr/include/c++/8/bits/stl_relops.h" 3
template< class _Tp> inline bool 
# 87
operator!=(const _Tp &__x, const _Tp &__y) 
# 88
{ return !(__x == __y); } 
# 98 "/usr/include/c++/8/bits/stl_relops.h" 3
template< class _Tp> inline bool 
# 100
operator>(const _Tp &__x, const _Tp &__y) 
# 101
{ return __y < __x; } 
# 111 "/usr/include/c++/8/bits/stl_relops.h" 3
template< class _Tp> inline bool 
# 113
operator<=(const _Tp &__x, const _Tp &__y) 
# 114
{ return !(__y < __x); } 
# 124 "/usr/include/c++/8/bits/stl_relops.h" 3
template< class _Tp> inline bool 
# 126
operator>=(const _Tp &__x, const _Tp &__y) 
# 127
{ return !(__x < __y); } 
# 128
}
# 131
}
# 36 "/usr/include/c++/8/bits/move.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 45
template< class _Tp> constexpr _Tp *
# 47
__addressof(_Tp &__r) noexcept 
# 48
{ return __builtin_addressof(__r); } 
# 53
}
# 40 "/usr/include/c++/8/type_traits" 3
namespace std __attribute((__visibility__("default"))) { 
# 56 "/usr/include/c++/8/type_traits" 3
template< class _Tp, _Tp __v> 
# 57
struct integral_constant { 
# 59
static constexpr _Tp value = (__v); 
# 60
typedef _Tp value_type; 
# 61
typedef integral_constant type; 
# 62
constexpr operator value_type() const noexcept { return value; } 
# 67
constexpr value_type operator()() const noexcept { return value; } 
# 69
}; 
# 71
template< class _Tp, _Tp __v> constexpr _Tp integral_constant< _Tp, __v> ::value; 
# 75
typedef integral_constant< bool, true>  true_type; 
# 78
typedef integral_constant< bool, false>  false_type; 
# 80
template< bool __v> using __bool_constant = integral_constant< bool, __v> ; 
# 91 "/usr/include/c++/8/type_traits" 3
template< bool , class , class > struct conditional; 
# 94
template< class ...> struct __or_; 
# 98
template<> struct __or_< >  : public false_type { 
# 100
}; 
# 102
template< class _B1> 
# 103
struct __or_< _B1>  : public _B1 { 
# 105
}; 
# 107
template< class _B1, class _B2> 
# 108
struct __or_< _B1, _B2>  : public conditional< _B1::value, _B1, _B2> ::type { 
# 110
}; 
# 112
template< class _B1, class _B2, class _B3, class ..._Bn> 
# 113
struct __or_< _B1, _B2, _B3, _Bn...>  : public conditional< _B1::value, _B1, std::__or_< _B2, _B3, _Bn...> > ::type { 
# 115
}; 
# 117
template< class ...> struct __and_; 
# 121
template<> struct __and_< >  : public true_type { 
# 123
}; 
# 125
template< class _B1> 
# 126
struct __and_< _B1>  : public _B1 { 
# 128
}; 
# 130
template< class _B1, class _B2> 
# 131
struct __and_< _B1, _B2>  : public conditional< _B1::value, _B2, _B1> ::type { 
# 133
}; 
# 135
template< class _B1, class _B2, class _B3, class ..._Bn> 
# 136
struct __and_< _B1, _B2, _B3, _Bn...>  : public conditional< _B1::value, std::__and_< _B2, _B3, _Bn...> , _B1> ::type { 
# 138
}; 
# 140
template< class _Pp> 
# 141
struct __not_ : public __bool_constant< !((bool)_Pp::value)>  { 
# 143
}; 
# 180 "/usr/include/c++/8/type_traits" 3
template< class _Tp> 
# 181
struct __success_type { 
# 182
typedef _Tp type; }; 
# 184
struct __failure_type { 
# 185
}; 
# 189
template< class > struct remove_cv; 
# 192
template< class > 
# 193
struct __is_void_helper : public false_type { 
# 194
}; 
# 197
template<> struct __is_void_helper< void>  : public true_type { 
# 198
}; 
# 201
template< class _Tp> 
# 202
struct is_void : public __is_void_helper< typename remove_cv< _Tp> ::type> ::type { 
# 204
}; 
# 206
template< class > 
# 207
struct __is_integral_helper : public false_type { 
# 208
}; 
# 211
template<> struct __is_integral_helper< bool>  : public true_type { 
# 212
}; 
# 215
template<> struct __is_integral_helper< char>  : public true_type { 
# 216
}; 
# 219
template<> struct __is_integral_helper< signed char>  : public true_type { 
# 220
}; 
# 223
template<> struct __is_integral_helper< unsigned char>  : public true_type { 
# 224
}; 
# 228
template<> struct __is_integral_helper< wchar_t>  : public true_type { 
# 229
}; 
# 233
template<> struct __is_integral_helper< char16_t>  : public true_type { 
# 234
}; 
# 237
template<> struct __is_integral_helper< char32_t>  : public true_type { 
# 238
}; 
# 241
template<> struct __is_integral_helper< short>  : public true_type { 
# 242
}; 
# 245
template<> struct __is_integral_helper< unsigned short>  : public true_type { 
# 246
}; 
# 249
template<> struct __is_integral_helper< int>  : public true_type { 
# 250
}; 
# 253
template<> struct __is_integral_helper< unsigned>  : public true_type { 
# 254
}; 
# 257
template<> struct __is_integral_helper< long>  : public true_type { 
# 258
}; 
# 261
template<> struct __is_integral_helper< unsigned long>  : public true_type { 
# 262
}; 
# 265
template<> struct __is_integral_helper< long long>  : public true_type { 
# 266
}; 
# 269
template<> struct __is_integral_helper< unsigned long long>  : public true_type { 
# 270
}; 
# 276
template<> struct __is_integral_helper< __int128>  : public true_type { 
# 277
}; 
# 280
template<> struct __is_integral_helper< unsigned __int128>  : public true_type { 
# 281
}; 
# 312 "/usr/include/c++/8/type_traits" 3
template< class _Tp> 
# 313
struct is_integral : public __is_integral_helper< typename remove_cv< _Tp> ::type> ::type { 
# 315
}; 
# 317
template< class > 
# 318
struct __is_floating_point_helper : public false_type { 
# 319
}; 
# 322
template<> struct __is_floating_point_helper< float>  : public true_type { 
# 323
}; 
# 326
template<> struct __is_floating_point_helper< double>  : public true_type { 
# 327
}; 
# 330
template<> struct __is_floating_point_helper< long double>  : public true_type { 
# 331
}; 
# 335
template<> struct __is_floating_point_helper< __float128>  : public true_type { 
# 336
}; 
# 340
template< class _Tp> 
# 341
struct is_floating_point : public __is_floating_point_helper< typename remove_cv< _Tp> ::type> ::type { 
# 343
}; 
# 346
template< class > 
# 347
struct is_array : public false_type { 
# 348
}; 
# 350
template< class _Tp, size_t _Size> 
# 351
struct is_array< _Tp [_Size]>  : public true_type { 
# 352
}; 
# 354
template< class _Tp> 
# 355
struct is_array< _Tp []>  : public true_type { 
# 356
}; 
# 358
template< class > 
# 359
struct __is_pointer_helper : public false_type { 
# 360
}; 
# 362
template< class _Tp> 
# 363
struct __is_pointer_helper< _Tp *>  : public true_type { 
# 364
}; 
# 367
template< class _Tp> 
# 368
struct is_pointer : public __is_pointer_helper< typename remove_cv< _Tp> ::type> ::type { 
# 370
}; 
# 373
template< class > 
# 374
struct is_lvalue_reference : public false_type { 
# 375
}; 
# 377
template< class _Tp> 
# 378
struct is_lvalue_reference< _Tp &>  : public true_type { 
# 379
}; 
# 382
template< class > 
# 383
struct is_rvalue_reference : public false_type { 
# 384
}; 
# 386
template< class _Tp> 
# 387
struct is_rvalue_reference< _Tp &&>  : public true_type { 
# 388
}; 
# 390
template< class > struct is_function; 
# 393
template< class > 
# 394
struct __is_member_object_pointer_helper : public false_type { 
# 395
}; 
# 397
template< class _Tp, class _Cp> 
# 398
struct __is_member_object_pointer_helper< _Tp (_Cp::*)>  : public integral_constant< bool, !is_function< _Tp> ::value>  { 
# 399
}; 
# 402
template< class _Tp> 
# 403
struct is_member_object_pointer : public __is_member_object_pointer_helper< typename remove_cv< _Tp> ::type> ::type { 
# 406
}; 
# 408
template< class > 
# 409
struct __is_member_function_pointer_helper : public false_type { 
# 410
}; 
# 412
template< class _Tp, class _Cp> 
# 413
struct __is_member_function_pointer_helper< _Tp (_Cp::*)>  : public integral_constant< bool, is_function< _Tp> ::value>  { 
# 414
}; 
# 417
template< class _Tp> 
# 418
struct is_member_function_pointer : public __is_member_function_pointer_helper< typename remove_cv< _Tp> ::type> ::type { 
# 421
}; 
# 424
template< class _Tp> 
# 425
struct is_enum : public integral_constant< bool, __is_enum(_Tp)>  { 
# 427
}; 
# 430
template< class _Tp> 
# 431
struct is_union : public integral_constant< bool, __is_union(_Tp)>  { 
# 433
}; 
# 436
template< class _Tp> 
# 437
struct is_class : public integral_constant< bool, __is_class(_Tp)>  { 
# 439
}; 
# 442
template< class > 
# 443
struct is_function : public false_type { 
# 444
}; 
# 446
template< class _Res, class ..._ArgTypes> 
# 447
struct is_function< _Res (_ArgTypes ...)>  : public true_type { 
# 448
}; 
# 450
template< class _Res, class ..._ArgTypes> 
# 451
struct is_function< _Res (_ArgTypes ...) &>  : public true_type { 
# 452
}; 
# 454
template< class _Res, class ..._ArgTypes> 
# 455
struct is_function< _Res (_ArgTypes ...) &&>  : public true_type { 
# 456
}; 
# 458
template< class _Res, class ..._ArgTypes> 
# 459
struct is_function< _Res (_ArgTypes ..., ...)>  : public true_type { 
# 460
}; 
# 462
template< class _Res, class ..._ArgTypes> 
# 463
struct is_function< _Res (_ArgTypes ..., ...) &>  : public true_type { 
# 464
}; 
# 466
template< class _Res, class ..._ArgTypes> 
# 467
struct is_function< _Res (_ArgTypes ..., ...) &&>  : public true_type { 
# 468
}; 
# 470
template< class _Res, class ..._ArgTypes> 
# 471
struct is_function< _Res (_ArgTypes ...) const>  : public true_type { 
# 472
}; 
# 474
template< class _Res, class ..._ArgTypes> 
# 475
struct is_function< _Res (_ArgTypes ...) const &>  : public true_type { 
# 476
}; 
# 478
template< class _Res, class ..._ArgTypes> 
# 479
struct is_function< _Res (_ArgTypes ...) const &&>  : public true_type { 
# 480
}; 
# 482
template< class _Res, class ..._ArgTypes> 
# 483
struct is_function< _Res (_ArgTypes ..., ...) const>  : public true_type { 
# 484
}; 
# 486
template< class _Res, class ..._ArgTypes> 
# 487
struct is_function< _Res (_ArgTypes ..., ...) const &>  : public true_type { 
# 488
}; 
# 490
template< class _Res, class ..._ArgTypes> 
# 491
struct is_function< _Res (_ArgTypes ..., ...) const &&>  : public true_type { 
# 492
}; 
# 494
template< class _Res, class ..._ArgTypes> 
# 495
struct is_function< _Res (_ArgTypes ...) volatile>  : public true_type { 
# 496
}; 
# 498
template< class _Res, class ..._ArgTypes> 
# 499
struct is_function< _Res (_ArgTypes ...) volatile &>  : public true_type { 
# 500
}; 
# 502
template< class _Res, class ..._ArgTypes> 
# 503
struct is_function< _Res (_ArgTypes ...) volatile &&>  : public true_type { 
# 504
}; 
# 506
template< class _Res, class ..._ArgTypes> 
# 507
struct is_function< _Res (_ArgTypes ..., ...) volatile>  : public true_type { 
# 508
}; 
# 510
template< class _Res, class ..._ArgTypes> 
# 511
struct is_function< _Res (_ArgTypes ..., ...) volatile &>  : public true_type { 
# 512
}; 
# 514
template< class _Res, class ..._ArgTypes> 
# 515
struct is_function< _Res (_ArgTypes ..., ...) volatile &&>  : public true_type { 
# 516
}; 
# 518
template< class _Res, class ..._ArgTypes> 
# 519
struct is_function< _Res (_ArgTypes ...) const volatile>  : public true_type { 
# 520
}; 
# 522
template< class _Res, class ..._ArgTypes> 
# 523
struct is_function< _Res (_ArgTypes ...) const volatile &>  : public true_type { 
# 524
}; 
# 526
template< class _Res, class ..._ArgTypes> 
# 527
struct is_function< _Res (_ArgTypes ...) const volatile &&>  : public true_type { 
# 528
}; 
# 530
template< class _Res, class ..._ArgTypes> 
# 531
struct is_function< _Res (_ArgTypes ..., ...) const volatile>  : public true_type { 
# 532
}; 
# 534
template< class _Res, class ..._ArgTypes> 
# 535
struct is_function< _Res (_ArgTypes ..., ...) const volatile &>  : public true_type { 
# 536
}; 
# 538
template< class _Res, class ..._ArgTypes> 
# 539
struct is_function< _Res (_ArgTypes ..., ...) const volatile &&>  : public true_type { 
# 540
}; 
# 544
template< class > 
# 545
struct __is_null_pointer_helper : public false_type { 
# 546
}; 
# 549
template<> struct __is_null_pointer_helper< nullptr_t>  : public true_type { 
# 550
}; 
# 553
template< class _Tp> 
# 554
struct is_null_pointer : public __is_null_pointer_helper< typename remove_cv< _Tp> ::type> ::type { 
# 556
}; 
# 559
template< class _Tp> 
# 560
struct __is_nullptr_t : public is_null_pointer< _Tp>  { 
# 562
}; 
# 567
template< class _Tp> 
# 568
struct is_reference : public __or_< is_lvalue_reference< _Tp> , is_rvalue_reference< _Tp> > ::type { 
# 571
}; 
# 574
template< class _Tp> 
# 575
struct is_arithmetic : public __or_< is_integral< _Tp> , is_floating_point< _Tp> > ::type { 
# 577
}; 
# 580
template< class _Tp> 
# 581
struct is_fundamental : public __or_< is_arithmetic< _Tp> , is_void< _Tp> , is_null_pointer< _Tp> > ::type { 
# 584
}; 
# 587
template< class _Tp> 
# 588
struct is_object : public __not_< __or_< is_function< _Tp> , is_reference< _Tp> , is_void< _Tp> > > ::type { 
# 591
}; 
# 593
template< class > struct is_member_pointer; 
# 597
template< class _Tp> 
# 598
struct is_scalar : public __or_< is_arithmetic< _Tp> , is_enum< _Tp> , is_pointer< _Tp> , is_member_pointer< _Tp> , is_null_pointer< _Tp> > ::type { 
# 601
}; 
# 604
template< class _Tp> 
# 605
struct is_compound : public integral_constant< bool, !is_fundamental< _Tp> ::value>  { 
# 606
}; 
# 608
template< class _Tp> 
# 609
struct __is_member_pointer_helper : public false_type { 
# 610
}; 
# 612
template< class _Tp, class _Cp> 
# 613
struct __is_member_pointer_helper< _Tp (_Cp::*)>  : public true_type { 
# 614
}; 
# 617
template< class _Tp> 
# 618
struct is_member_pointer : public __is_member_pointer_helper< typename remove_cv< _Tp> ::type> ::type { 
# 620
}; 
# 624
template< class _Tp> 
# 625
struct __is_referenceable : public __or_< is_object< _Tp> , is_reference< _Tp> > ::type { 
# 627
}; 
# 629
template< class _Res, class ..._Args> 
# 630
struct __is_referenceable< _Res (_Args ...)>  : public true_type { 
# 632
}; 
# 634
template< class _Res, class ..._Args> 
# 635
struct __is_referenceable< _Res (_Args ..., ...)>  : public true_type { 
# 637
}; 
# 642
template< class > 
# 643
struct is_const : public false_type { 
# 644
}; 
# 646
template< class _Tp> 
# 647
struct is_const< const _Tp>  : public true_type { 
# 648
}; 
# 651
template< class > 
# 652
struct is_volatile : public false_type { 
# 653
}; 
# 655
template< class _Tp> 
# 656
struct is_volatile< volatile _Tp>  : public true_type { 
# 657
}; 
# 660
template< class _Tp> 
# 661
struct is_trivial : public integral_constant< bool, __is_trivial(_Tp)>  { 
# 663
}; 
# 666
template< class _Tp> 
# 667
struct is_trivially_copyable : public integral_constant< bool, __is_trivially_copyable(_Tp)>  { 
# 669
}; 
# 672
template< class _Tp> 
# 673
struct is_standard_layout : public integral_constant< bool, __is_standard_layout(_Tp)>  { 
# 675
}; 
# 679
template< class _Tp> 
# 680
struct is_pod : public integral_constant< bool, __is_pod(_Tp)>  { 
# 682
}; 
# 685
template< class _Tp> 
# 686
struct is_literal_type : public integral_constant< bool, __is_literal_type(_Tp)>  { 
# 688
}; 
# 691
template< class _Tp> 
# 692
struct is_empty : public integral_constant< bool, __is_empty(_Tp)>  { 
# 694
}; 
# 697
template< class _Tp> 
# 698
struct is_polymorphic : public integral_constant< bool, __is_polymorphic(_Tp)>  { 
# 700
}; 
# 705
template< class _Tp> 
# 706
struct is_final : public integral_constant< bool, __is_final(_Tp)>  { 
# 708
}; 
# 712
template< class _Tp> 
# 713
struct is_abstract : public integral_constant< bool, __is_abstract(_Tp)>  { 
# 715
}; 
# 717
template< class _Tp, bool 
# 718
 = is_arithmetic< _Tp> ::value> 
# 719
struct __is_signed_helper : public false_type { 
# 720
}; 
# 722
template< class _Tp> 
# 723
struct __is_signed_helper< _Tp, true>  : public integral_constant< bool, ((_Tp)(-1)) < ((_Tp)0)>  { 
# 725
}; 
# 728
template< class _Tp> 
# 729
struct is_signed : public __is_signed_helper< _Tp> ::type { 
# 731
}; 
# 734
template< class _Tp> 
# 735
struct is_unsigned : public __and_< is_arithmetic< _Tp> , __not_< is_signed< _Tp> > >  { 
# 737
}; 
# 747 "/usr/include/c++/8/type_traits" 3
template< class _Tp, class _Up = _Tp &&> _Up __declval(int); 
# 751
template< class _Tp> _Tp __declval(long); 
# 755
template< class _Tp> auto declval() noexcept->__decltype((__declval< _Tp> (0))); 
# 758
template< class , unsigned  = 0U> struct extent; 
# 761
template< class > struct remove_all_extents; 
# 764
template< class _Tp> 
# 765
struct __is_array_known_bounds : public integral_constant< bool, (extent< _Tp> ::value > 0)>  { 
# 767
}; 
# 769
template< class _Tp> 
# 770
struct __is_array_unknown_bounds : public __and_< is_array< _Tp> , __not_< extent< _Tp> > >  { 
# 772
}; 
# 779
struct __do_is_destructible_impl { 
# 781
template< class _Tp, class  = __decltype((declval< _Tp &> ().~_Tp()))> static true_type __test(int); 
# 784
template< class > static false_type __test(...); 
# 786
}; 
# 788
template< class _Tp> 
# 789
struct __is_destructible_impl : public __do_is_destructible_impl { 
# 792
typedef __decltype((__test< _Tp> (0))) type; 
# 793
}; 
# 795
template< class _Tp, bool 
# 796
 = __or_< is_void< _Tp> , __is_array_unknown_bounds< _Tp> , is_function< _Tp> > ::value, bool 
# 799
 = __or_< is_reference< _Tp> , is_scalar< _Tp> > ::value> struct __is_destructible_safe; 
# 802
template< class _Tp> 
# 803
struct __is_destructible_safe< _Tp, false, false>  : public __is_destructible_impl< typename remove_all_extents< _Tp> ::type> ::type { 
# 806
}; 
# 808
template< class _Tp> 
# 809
struct __is_destructible_safe< _Tp, true, false>  : public false_type { 
# 810
}; 
# 812
template< class _Tp> 
# 813
struct __is_destructible_safe< _Tp, false, true>  : public true_type { 
# 814
}; 
# 817
template< class _Tp> 
# 818
struct is_destructible : public __is_destructible_safe< _Tp> ::type { 
# 820
}; 
# 826
struct __do_is_nt_destructible_impl { 
# 828
template< class _Tp> static integral_constant< bool, noexcept(declval< _Tp &> ().~_Tp())>  __test(int); 
# 832
template< class > static false_type __test(...); 
# 834
}; 
# 836
template< class _Tp> 
# 837
struct __is_nt_destructible_impl : public __do_is_nt_destructible_impl { 
# 840
typedef __decltype((__test< _Tp> (0))) type; 
# 841
}; 
# 843
template< class _Tp, bool 
# 844
 = __or_< is_void< _Tp> , __is_array_unknown_bounds< _Tp> , is_function< _Tp> > ::value, bool 
# 847
 = __or_< is_reference< _Tp> , is_scalar< _Tp> > ::value> struct __is_nt_destructible_safe; 
# 850
template< class _Tp> 
# 851
struct __is_nt_destructible_safe< _Tp, false, false>  : public __is_nt_destructible_impl< typename remove_all_extents< _Tp> ::type> ::type { 
# 854
}; 
# 856
template< class _Tp> 
# 857
struct __is_nt_destructible_safe< _Tp, true, false>  : public false_type { 
# 858
}; 
# 860
template< class _Tp> 
# 861
struct __is_nt_destructible_safe< _Tp, false, true>  : public true_type { 
# 862
}; 
# 865
template< class _Tp> 
# 866
struct is_nothrow_destructible : public __is_nt_destructible_safe< _Tp> ::type { 
# 868
}; 
# 871
template< class _Tp, class ..._Args> 
# 872
struct is_constructible : public __bool_constant< __is_constructible(_Tp, _Args...)>  { 
# 874
}; 
# 877
template< class _Tp> 
# 878
struct is_default_constructible : public is_constructible< _Tp> ::type { 
# 880
}; 
# 882
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_copy_constructible_impl; 
# 885
template< class _Tp> 
# 886
struct __is_copy_constructible_impl< _Tp, false>  : public false_type { 
# 887
}; 
# 889
template< class _Tp> 
# 890
struct __is_copy_constructible_impl< _Tp, true>  : public is_constructible< _Tp, const _Tp &>  { 
# 892
}; 
# 895
template< class _Tp> 
# 896
struct is_copy_constructible : public __is_copy_constructible_impl< _Tp>  { 
# 898
}; 
# 900
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_move_constructible_impl; 
# 903
template< class _Tp> 
# 904
struct __is_move_constructible_impl< _Tp, false>  : public false_type { 
# 905
}; 
# 907
template< class _Tp> 
# 908
struct __is_move_constructible_impl< _Tp, true>  : public is_constructible< _Tp, _Tp &&>  { 
# 910
}; 
# 913
template< class _Tp> 
# 914
struct is_move_constructible : public __is_move_constructible_impl< _Tp>  { 
# 916
}; 
# 918
template< bool , class _Tp, class ..._Args> 
# 919
struct __is_nt_constructible_impl : public false_type { 
# 921
}; 
# 923
template< class _Tp, class ..._Args> 
# 924
struct __is_nt_constructible_impl< true, _Tp, _Args...>  : public __bool_constant< noexcept((_Tp(std::declval< _Args> ()...)))>  { 
# 926
}; 
# 928
template< class _Tp, class _Arg> 
# 929
struct __is_nt_constructible_impl< true, _Tp, _Arg>  : public __bool_constant< noexcept((static_cast< _Tp>(std::declval< _Arg> ())))>  { 
# 931
}; 
# 933
template< class _Tp> 
# 934
struct __is_nt_constructible_impl< true, _Tp>  : public __bool_constant< noexcept((_Tp()))>  { 
# 936
}; 
# 938
template< class _Tp, size_t _Num> 
# 939
struct __is_nt_constructible_impl< true, _Tp [_Num]>  : public __bool_constant< noexcept((typename remove_all_extents< _Tp> ::type()))>  { 
# 941
}; 
# 943
template< class _Tp, class ..._Args> using __is_nothrow_constructible_impl = __is_nt_constructible_impl< __is_constructible(_Tp, _Args...), _Tp, _Args...> ; 
# 949
template< class _Tp, class ..._Args> 
# 950
struct is_nothrow_constructible : public __is_nt_constructible_impl< __is_constructible(_Tp, _Args...), _Tp, _Args...> ::type { 
# 952
}; 
# 955
template< class _Tp> 
# 956
struct is_nothrow_default_constructible : public __is_nt_constructible_impl< __is_constructible(_Tp), _Tp> ::type { 
# 958
}; 
# 961
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_nothrow_copy_constructible_impl; 
# 964
template< class _Tp> 
# 965
struct __is_nothrow_copy_constructible_impl< _Tp, false>  : public false_type { 
# 966
}; 
# 968
template< class _Tp> 
# 969
struct __is_nothrow_copy_constructible_impl< _Tp, true>  : public is_nothrow_constructible< _Tp, const _Tp &>  { 
# 971
}; 
# 974
template< class _Tp> 
# 975
struct is_nothrow_copy_constructible : public __is_nothrow_copy_constructible_impl< _Tp>  { 
# 977
}; 
# 979
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_nothrow_move_constructible_impl; 
# 982
template< class _Tp> 
# 983
struct __is_nothrow_move_constructible_impl< _Tp, false>  : public false_type { 
# 984
}; 
# 986
template< class _Tp> 
# 987
struct __is_nothrow_move_constructible_impl< _Tp, true>  : public is_nothrow_constructible< _Tp, _Tp &&>  { 
# 989
}; 
# 992
template< class _Tp> 
# 993
struct is_nothrow_move_constructible : public __is_nothrow_move_constructible_impl< _Tp>  { 
# 995
}; 
# 998
template< class _Tp, class _Up> 
# 999
struct is_assignable : public __bool_constant< __is_assignable(_Tp, _Up)>  { 
# 1001
}; 
# 1003
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_copy_assignable_impl; 
# 1006
template< class _Tp> 
# 1007
struct __is_copy_assignable_impl< _Tp, false>  : public false_type { 
# 1008
}; 
# 1010
template< class _Tp> 
# 1011
struct __is_copy_assignable_impl< _Tp, true>  : public is_assignable< _Tp &, const _Tp &>  { 
# 1013
}; 
# 1016
template< class _Tp> 
# 1017
struct is_copy_assignable : public __is_copy_assignable_impl< _Tp>  { 
# 1019
}; 
# 1021
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_move_assignable_impl; 
# 1024
template< class _Tp> 
# 1025
struct __is_move_assignable_impl< _Tp, false>  : public false_type { 
# 1026
}; 
# 1028
template< class _Tp> 
# 1029
struct __is_move_assignable_impl< _Tp, true>  : public is_assignable< _Tp &, _Tp &&>  { 
# 1031
}; 
# 1034
template< class _Tp> 
# 1035
struct is_move_assignable : public __is_move_assignable_impl< _Tp>  { 
# 1037
}; 
# 1039
template< class _Tp, class _Up> 
# 1040
struct __is_nt_assignable_impl : public integral_constant< bool, noexcept((declval< _Tp> () = declval< _Up> ()))>  { 
# 1042
}; 
# 1045
template< class _Tp, class _Up> 
# 1046
struct is_nothrow_assignable : public __and_< is_assignable< _Tp, _Up> , __is_nt_assignable_impl< _Tp, _Up> >  { 
# 1049
}; 
# 1051
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_nt_copy_assignable_impl; 
# 1054
template< class _Tp> 
# 1055
struct __is_nt_copy_assignable_impl< _Tp, false>  : public false_type { 
# 1056
}; 
# 1058
template< class _Tp> 
# 1059
struct __is_nt_copy_assignable_impl< _Tp, true>  : public is_nothrow_assignable< _Tp &, const _Tp &>  { 
# 1061
}; 
# 1064
template< class _Tp> 
# 1065
struct is_nothrow_copy_assignable : public __is_nt_copy_assignable_impl< _Tp>  { 
# 1067
}; 
# 1069
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_nt_move_assignable_impl; 
# 1072
template< class _Tp> 
# 1073
struct __is_nt_move_assignable_impl< _Tp, false>  : public false_type { 
# 1074
}; 
# 1076
template< class _Tp> 
# 1077
struct __is_nt_move_assignable_impl< _Tp, true>  : public is_nothrow_assignable< _Tp &, _Tp &&>  { 
# 1079
}; 
# 1082
template< class _Tp> 
# 1083
struct is_nothrow_move_assignable : public __is_nt_move_assignable_impl< _Tp>  { 
# 1085
}; 
# 1088
template< class _Tp, class ..._Args> 
# 1089
struct is_trivially_constructible : public __and_< is_constructible< _Tp, _Args...> , __bool_constant< __is_trivially_constructible(_Tp, _Args...)> > ::type { 
# 1092
}; 
# 1095
template< class _Tp> 
# 1096
struct is_trivially_default_constructible : public is_trivially_constructible< _Tp> ::type { 
# 1098
}; 
# 1100
struct __do_is_implicitly_default_constructible_impl { 
# 1102
template< class _Tp> static void __helper(const _Tp &); 
# 1105
template< class _Tp> static true_type __test(const _Tp &, __decltype((__helper< const _Tp &> ({}))) * = 0); 
# 1109
static false_type __test(...); 
# 1110
}; 
# 1112
template< class _Tp> 
# 1113
struct __is_implicitly_default_constructible_impl : public __do_is_implicitly_default_constructible_impl { 
# 1116
typedef __decltype((__test(declval< _Tp> ()))) type; 
# 1117
}; 
# 1119
template< class _Tp> 
# 1120
struct __is_implicitly_default_constructible_safe : public __is_implicitly_default_constructible_impl< _Tp> ::type { 
# 1122
}; 
# 1124
template< class _Tp> 
# 1125
struct __is_implicitly_default_constructible : public __and_< is_default_constructible< _Tp> , __is_implicitly_default_constructible_safe< _Tp> >  { 
# 1128
}; 
# 1132
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_trivially_copy_constructible_impl; 
# 1135
template< class _Tp> 
# 1136
struct __is_trivially_copy_constructible_impl< _Tp, false>  : public false_type { 
# 1137
}; 
# 1139
template< class _Tp> 
# 1140
struct __is_trivially_copy_constructible_impl< _Tp, true>  : public __and_< is_copy_constructible< _Tp> , integral_constant< bool, __is_trivially_constructible(_Tp, const _Tp &)> >  { 
# 1144
}; 
# 1146
template< class _Tp> 
# 1147
struct is_trivially_copy_constructible : public __is_trivially_copy_constructible_impl< _Tp>  { 
# 1149
}; 
# 1153
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_trivially_move_constructible_impl; 
# 1156
template< class _Tp> 
# 1157
struct __is_trivially_move_constructible_impl< _Tp, false>  : public false_type { 
# 1158
}; 
# 1160
template< class _Tp> 
# 1161
struct __is_trivially_move_constructible_impl< _Tp, true>  : public __and_< is_move_constructible< _Tp> , integral_constant< bool, __is_trivially_constructible(_Tp, _Tp &&)> >  { 
# 1165
}; 
# 1167
template< class _Tp> 
# 1168
struct is_trivially_move_constructible : public __is_trivially_move_constructible_impl< _Tp>  { 
# 1170
}; 
# 1173
template< class _Tp, class _Up> 
# 1174
struct is_trivially_assignable : public __bool_constant< __is_trivially_assignable(_Tp, _Up)>  { 
# 1176
}; 
# 1180
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_trivially_copy_assignable_impl; 
# 1183
template< class _Tp> 
# 1184
struct __is_trivially_copy_assignable_impl< _Tp, false>  : public false_type { 
# 1185
}; 
# 1187
template< class _Tp> 
# 1188
struct __is_trivially_copy_assignable_impl< _Tp, true>  : public __and_< is_copy_assignable< _Tp> , integral_constant< bool, __is_trivially_assignable(_Tp &, const _Tp &)> >  { 
# 1192
}; 
# 1194
template< class _Tp> 
# 1195
struct is_trivially_copy_assignable : public __is_trivially_copy_assignable_impl< _Tp>  { 
# 1197
}; 
# 1201
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_trivially_move_assignable_impl; 
# 1204
template< class _Tp> 
# 1205
struct __is_trivially_move_assignable_impl< _Tp, false>  : public false_type { 
# 1206
}; 
# 1208
template< class _Tp> 
# 1209
struct __is_trivially_move_assignable_impl< _Tp, true>  : public __and_< is_move_assignable< _Tp> , integral_constant< bool, __is_trivially_assignable(_Tp &, _Tp &&)> >  { 
# 1213
}; 
# 1215
template< class _Tp> 
# 1216
struct is_trivially_move_assignable : public __is_trivially_move_assignable_impl< _Tp>  { 
# 1218
}; 
# 1221
template< class _Tp> 
# 1222
struct is_trivially_destructible : public __and_< is_destructible< _Tp> , integral_constant< bool, __has_trivial_destructor(_Tp)> >  { 
# 1225
}; 
# 1229
template< class _Tp> 
# 1230
struct has_virtual_destructor : public integral_constant< bool, __has_virtual_destructor(_Tp)>  { 
# 1232
}; 
# 1238
template< class _Tp> 
# 1239
struct alignment_of : public integral_constant< unsigned long, __alignof__(_Tp)>  { 
# 1240
}; 
# 1243
template< class > 
# 1244
struct rank : public integral_constant< unsigned long, 0UL>  { 
# 1245
}; 
# 1247
template< class _Tp, size_t _Size> 
# 1248
struct rank< _Tp [_Size]>  : public integral_constant< unsigned long, 1 + std::rank< _Tp> ::value>  { 
# 1249
}; 
# 1251
template< class _Tp> 
# 1252
struct rank< _Tp []>  : public integral_constant< unsigned long, 1 + std::rank< _Tp> ::value>  { 
# 1253
}; 
# 1256
template< class , unsigned _Uint> 
# 1257
struct extent : public integral_constant< unsigned long, 0UL>  { 
# 1258
}; 
# 1260
template< class _Tp, unsigned _Uint, size_t _Size> 
# 1261
struct extent< _Tp [_Size], _Uint>  : public integral_constant< unsigned long, (_Uint == (0)) ? _Size : std::extent< _Tp, _Uint - (1)> ::value>  { 
# 1265
}; 
# 1267
template< class _Tp, unsigned _Uint> 
# 1268
struct extent< _Tp [], _Uint>  : public integral_constant< unsigned long, (_Uint == (0)) ? 0 : std::extent< _Tp, _Uint - (1)> ::value>  { 
# 1272
}; 
# 1278
template< class , class > 
# 1279
struct is_same : public false_type { 
# 1280
}; 
# 1282
template< class _Tp> 
# 1283
struct is_same< _Tp, _Tp>  : public true_type { 
# 1284
}; 
# 1287
template< class _Base, class _Derived> 
# 1288
struct is_base_of : public integral_constant< bool, __is_base_of(_Base, _Derived)>  { 
# 1290
}; 
# 1292
template< class _From, class _To, bool 
# 1293
 = __or_< is_void< _From> , is_function< _To> , is_array< _To> > ::value> 
# 1295
struct __is_convertible_helper { 
# 1296
typedef typename is_void< _To> ::type type; }; 
# 1298
template< class _From, class _To> 
# 1299
class __is_convertible_helper< _From, _To, false>  { 
# 1301
template< class _To1> static void __test_aux(_To1); 
# 1304
template< class _From1, class _To1, class 
# 1305
 = __decltype((__test_aux< _To1> (std::declval< _From1> ())))> static true_type 
# 1304
__test(int); 
# 1309
template< class , class > static false_type __test(...); 
# 1314
public: typedef __decltype((__test< _From, _To> (0))) type; 
# 1315
}; 
# 1319
template< class _From, class _To> 
# 1320
struct is_convertible : public __is_convertible_helper< _From, _To> ::type { 
# 1322
}; 
# 1328
template< class _Tp> 
# 1329
struct remove_const { 
# 1330
typedef _Tp type; }; 
# 1332
template< class _Tp> 
# 1333
struct remove_const< const _Tp>  { 
# 1334
typedef _Tp type; }; 
# 1337
template< class _Tp> 
# 1338
struct remove_volatile { 
# 1339
typedef _Tp type; }; 
# 1341
template< class _Tp> 
# 1342
struct remove_volatile< volatile _Tp>  { 
# 1343
typedef _Tp type; }; 
# 1346
template< class _Tp> 
# 1347
struct remove_cv { 
# 1350
typedef typename remove_const< typename remove_volatile< _Tp> ::type> ::type type; 
# 1351
}; 
# 1354
template< class _Tp> 
# 1355
struct add_const { 
# 1356
typedef const _Tp type; }; 
# 1359
template< class _Tp> 
# 1360
struct add_volatile { 
# 1361
typedef volatile _Tp type; }; 
# 1364
template< class _Tp> 
# 1365
struct add_cv { 
# 1368
typedef typename add_const< typename add_volatile< _Tp> ::type> ::type type; 
# 1369
}; 
# 1376
template< class _Tp> using remove_const_t = typename remove_const< _Tp> ::type; 
# 1380
template< class _Tp> using remove_volatile_t = typename remove_volatile< _Tp> ::type; 
# 1384
template< class _Tp> using remove_cv_t = typename remove_cv< _Tp> ::type; 
# 1388
template< class _Tp> using add_const_t = typename add_const< _Tp> ::type; 
# 1392
template< class _Tp> using add_volatile_t = typename add_volatile< _Tp> ::type; 
# 1396
template< class _Tp> using add_cv_t = typename add_cv< _Tp> ::type; 
# 1403
template< class _Tp> 
# 1404
struct remove_reference { 
# 1405
typedef _Tp type; }; 
# 1407
template< class _Tp> 
# 1408
struct remove_reference< _Tp &>  { 
# 1409
typedef _Tp type; }; 
# 1411
template< class _Tp> 
# 1412
struct remove_reference< _Tp &&>  { 
# 1413
typedef _Tp type; }; 
# 1415
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> 
# 1416
struct __add_lvalue_reference_helper { 
# 1417
typedef _Tp type; }; 
# 1419
template< class _Tp> 
# 1420
struct __add_lvalue_reference_helper< _Tp, true>  { 
# 1421
typedef _Tp &type; }; 
# 1424
template< class _Tp> 
# 1425
struct add_lvalue_reference : public __add_lvalue_reference_helper< _Tp>  { 
# 1427
}; 
# 1429
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> 
# 1430
struct __add_rvalue_reference_helper { 
# 1431
typedef _Tp type; }; 
# 1433
template< class _Tp> 
# 1434
struct __add_rvalue_reference_helper< _Tp, true>  { 
# 1435
typedef _Tp &&type; }; 
# 1438
template< class _Tp> 
# 1439
struct add_rvalue_reference : public __add_rvalue_reference_helper< _Tp>  { 
# 1441
}; 
# 1445
template< class _Tp> using remove_reference_t = typename remove_reference< _Tp> ::type; 
# 1449
template< class _Tp> using add_lvalue_reference_t = typename add_lvalue_reference< _Tp> ::type; 
# 1453
template< class _Tp> using add_rvalue_reference_t = typename add_rvalue_reference< _Tp> ::type; 
# 1460
template< class _Unqualified, bool _IsConst, bool _IsVol> struct __cv_selector; 
# 1463
template< class _Unqualified> 
# 1464
struct __cv_selector< _Unqualified, false, false>  { 
# 1465
typedef _Unqualified __type; }; 
# 1467
template< class _Unqualified> 
# 1468
struct __cv_selector< _Unqualified, false, true>  { 
# 1469
typedef volatile _Unqualified __type; }; 
# 1471
template< class _Unqualified> 
# 1472
struct __cv_selector< _Unqualified, true, false>  { 
# 1473
typedef const _Unqualified __type; }; 
# 1475
template< class _Unqualified> 
# 1476
struct __cv_selector< _Unqualified, true, true>  { 
# 1477
typedef const volatile _Unqualified __type; }; 
# 1479
template< class _Qualified, class _Unqualified, bool 
# 1480
_IsConst = is_const< _Qualified> ::value, bool 
# 1481
_IsVol = is_volatile< _Qualified> ::value> 
# 1482
class __match_cv_qualifiers { 
# 1484
typedef __cv_selector< _Unqualified, _IsConst, _IsVol>  __match; 
# 1487
public: typedef typename __cv_selector< _Unqualified, _IsConst, _IsVol> ::__type __type; 
# 1488
}; 
# 1491
template< class _Tp> 
# 1492
struct __make_unsigned { 
# 1493
typedef _Tp __type; }; 
# 1496
template<> struct __make_unsigned< char>  { 
# 1497
typedef unsigned char __type; }; 
# 1500
template<> struct __make_unsigned< signed char>  { 
# 1501
typedef unsigned char __type; }; 
# 1504
template<> struct __make_unsigned< short>  { 
# 1505
typedef unsigned short __type; }; 
# 1508
template<> struct __make_unsigned< int>  { 
# 1509
typedef unsigned __type; }; 
# 1512
template<> struct __make_unsigned< long>  { 
# 1513
typedef unsigned long __type; }; 
# 1516
template<> struct __make_unsigned< long long>  { 
# 1517
typedef unsigned long long __type; }; 
# 1521
template<> struct __make_unsigned< __int128>  { 
# 1522
typedef unsigned __int128 __type; }; 
# 1541 "/usr/include/c++/8/type_traits" 3
template< class _Tp, bool 
# 1542
_IsInt = is_integral< _Tp> ::value, bool 
# 1543
_IsEnum = is_enum< _Tp> ::value> class __make_unsigned_selector; 
# 1546
template< class _Tp> 
# 1547
class __make_unsigned_selector< _Tp, true, false>  { 
# 1549
typedef __make_unsigned< typename remove_cv< _Tp> ::type>  __unsignedt; 
# 1550
typedef typename __make_unsigned< typename remove_cv< _Tp> ::type> ::__type __unsigned_type; 
# 1551
typedef __match_cv_qualifiers< _Tp, __unsigned_type>  __cv_unsigned; 
# 1554
public: typedef typename __match_cv_qualifiers< _Tp, __unsigned_type> ::__type __type; 
# 1555
}; 
# 1557
template< class _Tp> 
# 1558
class __make_unsigned_selector< _Tp, false, true>  { 
# 1561
typedef unsigned char __smallest; 
# 1562
static const bool __b0 = (sizeof(_Tp) <= sizeof(__smallest)); 
# 1563
static const bool __b1 = (sizeof(_Tp) <= sizeof(unsigned short)); 
# 1564
static const bool __b2 = (sizeof(_Tp) <= sizeof(unsigned)); 
# 1565
static const bool __b3 = (sizeof(_Tp) <= sizeof(unsigned long)); 
# 1566
typedef conditional< __b3, unsigned long, unsigned long long>  __cond3; 
# 1567
typedef typename conditional< __b3, unsigned long, unsigned long long> ::type __cond3_type; 
# 1568
typedef conditional< __b2, unsigned, __cond3_type>  __cond2; 
# 1569
typedef typename conditional< __b2, unsigned, __cond3_type> ::type __cond2_type; 
# 1570
typedef conditional< __b1, unsigned short, __cond2_type>  __cond1; 
# 1571
typedef typename conditional< __b1, unsigned short, __cond2_type> ::type __cond1_type; 
# 1574
typedef typename conditional< __b0, unsigned char, __cond1_type> ::type __unsigned_type; 
# 1575
typedef __match_cv_qualifiers< _Tp, __unsigned_type>  __cv_unsigned; 
# 1578
public: typedef typename __match_cv_qualifiers< _Tp, __unsigned_type> ::__type __type; 
# 1579
}; 
# 1585
template< class _Tp> 
# 1586
struct make_unsigned { 
# 1587
typedef typename __make_unsigned_selector< _Tp> ::__type type; }; 
# 1591
template<> struct make_unsigned< bool> ; 
# 1595
template< class _Tp> 
# 1596
struct __make_signed { 
# 1597
typedef _Tp __type; }; 
# 1600
template<> struct __make_signed< char>  { 
# 1601
typedef signed char __type; }; 
# 1604
template<> struct __make_signed< unsigned char>  { 
# 1605
typedef signed char __type; }; 
# 1608
template<> struct __make_signed< unsigned short>  { 
# 1609
typedef signed short __type; }; 
# 1612
template<> struct __make_signed< unsigned>  { 
# 1613
typedef signed int __type; }; 
# 1616
template<> struct __make_signed< unsigned long>  { 
# 1617
typedef signed long __type; }; 
# 1620
template<> struct __make_signed< unsigned long long>  { 
# 1621
typedef signed long long __type; }; 
# 1625
template<> struct __make_signed< unsigned __int128>  { 
# 1626
typedef __int128 __type; }; 
# 1645 "/usr/include/c++/8/type_traits" 3
template< class _Tp, bool 
# 1646
_IsInt = is_integral< _Tp> ::value, bool 
# 1647
_IsEnum = is_enum< _Tp> ::value> class __make_signed_selector; 
# 1650
template< class _Tp> 
# 1651
class __make_signed_selector< _Tp, true, false>  { 
# 1653
typedef __make_signed< typename remove_cv< _Tp> ::type>  __signedt; 
# 1654
typedef typename __make_signed< typename remove_cv< _Tp> ::type> ::__type __signed_type; 
# 1655
typedef __match_cv_qualifiers< _Tp, __signed_type>  __cv_signed; 
# 1658
public: typedef typename __match_cv_qualifiers< _Tp, __signed_type> ::__type __type; 
# 1659
}; 
# 1661
template< class _Tp> 
# 1662
class __make_signed_selector< _Tp, false, true>  { 
# 1664
typedef typename __make_unsigned_selector< _Tp> ::__type __unsigned_type; 
# 1667
public: typedef typename std::__make_signed_selector< __unsigned_type> ::__type __type; 
# 1668
}; 
# 1674
template< class _Tp> 
# 1675
struct make_signed { 
# 1676
typedef typename __make_signed_selector< _Tp> ::__type type; }; 
# 1680
template<> struct make_signed< bool> ; 
# 1684
template< class _Tp> using make_signed_t = typename make_signed< _Tp> ::type; 
# 1688
template< class _Tp> using make_unsigned_t = typename make_unsigned< _Tp> ::type; 
# 1695
template< class _Tp> 
# 1696
struct remove_extent { 
# 1697
typedef _Tp type; }; 
# 1699
template< class _Tp, size_t _Size> 
# 1700
struct remove_extent< _Tp [_Size]>  { 
# 1701
typedef _Tp type; }; 
# 1703
template< class _Tp> 
# 1704
struct remove_extent< _Tp []>  { 
# 1705
typedef _Tp type; }; 
# 1708
template< class _Tp> 
# 1709
struct remove_all_extents { 
# 1710
typedef _Tp type; }; 
# 1712
template< class _Tp, size_t _Size> 
# 1713
struct remove_all_extents< _Tp [_Size]>  { 
# 1714
typedef typename std::remove_all_extents< _Tp> ::type type; }; 
# 1716
template< class _Tp> 
# 1717
struct remove_all_extents< _Tp []>  { 
# 1718
typedef typename std::remove_all_extents< _Tp> ::type type; }; 
# 1722
template< class _Tp> using remove_extent_t = typename remove_extent< _Tp> ::type; 
# 1726
template< class _Tp> using remove_all_extents_t = typename remove_all_extents< _Tp> ::type; 
# 1732
template< class _Tp, class > 
# 1733
struct __remove_pointer_helper { 
# 1734
typedef _Tp type; }; 
# 1736
template< class _Tp, class _Up> 
# 1737
struct __remove_pointer_helper< _Tp, _Up *>  { 
# 1738
typedef _Up type; }; 
# 1741
template< class _Tp> 
# 1742
struct remove_pointer : public __remove_pointer_helper< _Tp, typename remove_cv< _Tp> ::type>  { 
# 1744
}; 
# 1747
template< class _Tp, bool  = __or_< __is_referenceable< _Tp> , is_void< _Tp> > ::value> 
# 1749
struct __add_pointer_helper { 
# 1750
typedef _Tp type; }; 
# 1752
template< class _Tp> 
# 1753
struct __add_pointer_helper< _Tp, true>  { 
# 1754
typedef typename remove_reference< _Tp> ::type *type; }; 
# 1756
template< class _Tp> 
# 1757
struct add_pointer : public __add_pointer_helper< _Tp>  { 
# 1759
}; 
# 1763
template< class _Tp> using remove_pointer_t = typename remove_pointer< _Tp> ::type; 
# 1767
template< class _Tp> using add_pointer_t = typename add_pointer< _Tp> ::type; 
# 1771
template< size_t _Len> 
# 1772
struct __aligned_storage_msa { 
# 1774
union __type { 
# 1776
unsigned char __data[_Len]; 
# 1777
struct __attribute((__aligned__)) { } __align; 
# 1778
}; 
# 1779
}; 
# 1791 "/usr/include/c++/8/type_traits" 3
template< size_t _Len, size_t _Align = __alignof__(typename __aligned_storage_msa< _Len> ::__type)> 
# 1793
struct aligned_storage { 
# 1795
union type { 
# 1797
unsigned char __data[_Len]; 
# 1798
struct __attribute((__aligned__(_Align))) { } __align; 
# 1799
}; 
# 1800
}; 
# 1802
template< class ..._Types> 
# 1803
struct __strictest_alignment { 
# 1805
static const size_t _S_alignment = (0); 
# 1806
static const size_t _S_size = (0); 
# 1807
}; 
# 1809
template< class _Tp, class ..._Types> 
# 1810
struct __strictest_alignment< _Tp, _Types...>  { 
# 1812
static const size_t _S_alignment = ((__alignof__(_Tp) > __strictest_alignment< _Types...> ::_S_alignment) ? __alignof__(_Tp) : __strictest_alignment< _Types...> ::_S_alignment); 
# 1815
static const size_t _S_size = ((sizeof(_Tp) > __strictest_alignment< _Types...> ::_S_size) ? sizeof(_Tp) : __strictest_alignment< _Types...> ::_S_size); 
# 1818
}; 
# 1830 "/usr/include/c++/8/type_traits" 3
template< size_t _Len, class ..._Types> 
# 1831
struct aligned_union { 
# 1834
static_assert((sizeof...(_Types) != (0)), "At least one type is required");
# 1836
private: using __strictest = __strictest_alignment< _Types...> ; 
# 1837
static const size_t _S_len = ((_Len > __strictest::_S_size) ? _Len : __strictest::_S_size); 
# 1841
public: static const size_t alignment_value = (__strictest::_S_alignment); 
# 1843
typedef typename aligned_storage< _S_len, alignment_value> ::type type; 
# 1844
}; 
# 1846
template< size_t _Len, class ..._Types> const size_t aligned_union< _Len, _Types...> ::alignment_value; 
# 1851
template< class _Up, bool 
# 1852
_IsArray = is_array< _Up> ::value, bool 
# 1853
_IsFunction = is_function< _Up> ::value> struct __decay_selector; 
# 1857
template< class _Up> 
# 1858
struct __decay_selector< _Up, false, false>  { 
# 1859
typedef typename remove_cv< _Up> ::type __type; }; 
# 1861
template< class _Up> 
# 1862
struct __decay_selector< _Up, true, false>  { 
# 1863
typedef typename remove_extent< _Up> ::type *__type; }; 
# 1865
template< class _Up> 
# 1866
struct __decay_selector< _Up, false, true>  { 
# 1867
typedef typename add_pointer< _Up> ::type __type; }; 
# 1870
template< class _Tp> 
# 1871
class decay { 
# 1873
typedef typename remove_reference< _Tp> ::type __remove_type; 
# 1876
public: typedef typename __decay_selector< __remove_type> ::__type type; 
# 1877
}; 
# 1879
template< class _Tp> class reference_wrapper; 
# 1883
template< class _Tp> 
# 1884
struct __strip_reference_wrapper { 
# 1886
typedef _Tp __type; 
# 1887
}; 
# 1889
template< class _Tp> 
# 1890
struct __strip_reference_wrapper< reference_wrapper< _Tp> >  { 
# 1892
typedef _Tp &__type; 
# 1893
}; 
# 1895
template< class _Tp> 
# 1896
struct __decay_and_strip { 
# 1899
typedef typename __strip_reference_wrapper< typename decay< _Tp> ::type> ::__type __type; 
# 1900
}; 
# 1905
template< bool , class _Tp = void> 
# 1906
struct enable_if { 
# 1907
}; 
# 1910
template< class _Tp> 
# 1911
struct enable_if< true, _Tp>  { 
# 1912
typedef _Tp type; }; 
# 1914
template< class ..._Cond> using _Require = typename enable_if< __and_< _Cond...> ::value> ::type; 
# 1919
template< bool _Cond, class _Iftrue, class _Iffalse> 
# 1920
struct conditional { 
# 1921
typedef _Iftrue type; }; 
# 1924
template< class _Iftrue, class _Iffalse> 
# 1925
struct conditional< false, _Iftrue, _Iffalse>  { 
# 1926
typedef _Iffalse type; }; 
# 1929
template< class ..._Tp> struct common_type; 
# 1934
struct __do_common_type_impl { 
# 1936
template< class _Tp, class _Up> static __success_type< typename decay< __decltype((true ? std::declval< _Tp> () : std::declval< _Up> ()))> ::type>  _S_test(int); 
# 1941
template< class , class > static __failure_type _S_test(...); 
# 1943
}; 
# 1945
template< class _Tp, class _Up> 
# 1946
struct __common_type_impl : private __do_common_type_impl { 
# 1949
typedef __decltype((_S_test< _Tp, _Up> (0))) type; 
# 1950
}; 
# 1952
struct __do_member_type_wrapper { 
# 1954
template< class _Tp> static __success_type< typename _Tp::type>  _S_test(int); 
# 1957
template< class > static __failure_type _S_test(...); 
# 1959
}; 
# 1961
template< class _Tp> 
# 1962
struct __member_type_wrapper : private __do_member_type_wrapper { 
# 1965
typedef __decltype((_S_test< _Tp> (0))) type; 
# 1966
}; 
# 1968
template< class _CTp, class ..._Args> 
# 1969
struct __expanded_common_type_wrapper { 
# 1971
typedef common_type< typename _CTp::type, _Args...>  type; 
# 1972
}; 
# 1974
template< class ..._Args> 
# 1975
struct __expanded_common_type_wrapper< __failure_type, _Args...>  { 
# 1976
typedef __failure_type type; }; 
# 1979
template<> struct common_type< >  { 
# 1980
}; 
# 1982
template< class _Tp> 
# 1983
struct common_type< _Tp>  : public std::common_type< _Tp, _Tp>  { 
# 1985
}; 
# 1987
template< class _Tp, class _Up> 
# 1988
struct common_type< _Tp, _Up>  : public __common_type_impl< _Tp, _Up> ::type { 
# 1990
}; 
# 1992
template< class _Tp, class _Up, class ..._Vp> 
# 1993
struct common_type< _Tp, _Up, _Vp...>  : public __expanded_common_type_wrapper< typename __member_type_wrapper< std::common_type< _Tp, _Up> > ::type, _Vp...> ::type { 
# 1996
}; 
# 1999
template< class _Tp> 
# 2000
struct underlying_type { 
# 2002
typedef __underlying_type(_Tp) type; 
# 2003
}; 
# 2005
template< class _Tp> 
# 2006
struct __declval_protector { 
# 2008
static const bool __stop = false; 
# 2009
}; 
# 2011
template< class _Tp> auto 
# 2012
declval() noexcept->__decltype((__declval< _Tp> (0))) 
# 2013
{ 
# 2014
static_assert((__declval_protector< _Tp> ::__stop), "declval() must not be used!");
# 2016
return __declval< _Tp> (0); 
# 2017
} 
# 2029 "/usr/include/c++/8/type_traits" 3
template<> struct __make_unsigned< wchar_t>  { 
# 2031
using __type = __make_unsigned_selector< wchar_t, false, true> ::__type; 
# 2033
}; 
# 2036
template<> struct __make_signed< wchar_t>  { 
# 2038
using __type = __make_signed_selector< wchar_t, false, true> ::__type; 
# 2040
}; 
# 2044
template<> struct __make_unsigned< char16_t>  { 
# 2046
using __type = __make_unsigned_selector< char16_t, false, true> ::__type; 
# 2048
}; 
# 2051
template<> struct __make_signed< char16_t>  { 
# 2053
using __type = __make_signed_selector< char16_t, false, true> ::__type; 
# 2055
}; 
# 2058
template<> struct __make_unsigned< char32_t>  { 
# 2060
using __type = __make_unsigned_selector< char32_t, false, true> ::__type; 
# 2062
}; 
# 2065
template<> struct __make_signed< char32_t>  { 
# 2067
using __type = __make_signed_selector< char32_t, false, true> ::__type; 
# 2069
}; 
# 2073
template< class _Signature> class result_of; 
# 2080
struct __invoke_memfun_ref { }; 
# 2081
struct __invoke_memfun_deref { }; 
# 2082
struct __invoke_memobj_ref { }; 
# 2083
struct __invoke_memobj_deref { }; 
# 2084
struct __invoke_other { }; 
# 2087
template< class _Tp, class _Tag> 
# 2088
struct __result_of_success : public __success_type< _Tp>  { 
# 2089
using __invoke_type = _Tag; }; 
# 2092
struct __result_of_memfun_ref_impl { 
# 2094
template< class _Fp, class _Tp1, class ..._Args> static __result_of_success< __decltype(((std::declval< _Tp1> ().*std::declval< _Fp> ())(std::declval< _Args> ()...))), __invoke_memfun_ref>  _S_test(int); 
# 2099
template< class ...> static __failure_type _S_test(...); 
# 2101
}; 
# 2103
template< class _MemPtr, class _Arg, class ..._Args> 
# 2104
struct __result_of_memfun_ref : private __result_of_memfun_ref_impl { 
# 2107
typedef __decltype((_S_test< _MemPtr, _Arg, _Args...> (0))) type; 
# 2108
}; 
# 2111
struct __result_of_memfun_deref_impl { 
# 2113
template< class _Fp, class _Tp1, class ..._Args> static __result_of_success< __decltype((((*std::declval< _Tp1> ()).*std::declval< _Fp> ())(std::declval< _Args> ()...))), __invoke_memfun_deref>  _S_test(int); 
# 2118
template< class ...> static __failure_type _S_test(...); 
# 2120
}; 
# 2122
template< class _MemPtr, class _Arg, class ..._Args> 
# 2123
struct __result_of_memfun_deref : private __result_of_memfun_deref_impl { 
# 2126
typedef __decltype((_S_test< _MemPtr, _Arg, _Args...> (0))) type; 
# 2127
}; 
# 2130
struct __result_of_memobj_ref_impl { 
# 2132
template< class _Fp, class _Tp1> static __result_of_success< __decltype((std::declval< _Tp1> ().*std::declval< _Fp> ())), __invoke_memobj_ref>  _S_test(int); 
# 2137
template< class , class > static __failure_type _S_test(...); 
# 2139
}; 
# 2141
template< class _MemPtr, class _Arg> 
# 2142
struct __result_of_memobj_ref : private __result_of_memobj_ref_impl { 
# 2145
typedef __decltype((_S_test< _MemPtr, _Arg> (0))) type; 
# 2146
}; 
# 2149
struct __result_of_memobj_deref_impl { 
# 2151
template< class _Fp, class _Tp1> static __result_of_success< __decltype(((*std::declval< _Tp1> ()).*std::declval< _Fp> ())), __invoke_memobj_deref>  _S_test(int); 
# 2156
template< class , class > static __failure_type _S_test(...); 
# 2158
}; 
# 2160
template< class _MemPtr, class _Arg> 
# 2161
struct __result_of_memobj_deref : private __result_of_memobj_deref_impl { 
# 2164
typedef __decltype((_S_test< _MemPtr, _Arg> (0))) type; 
# 2165
}; 
# 2167
template< class _MemPtr, class _Arg> struct __result_of_memobj; 
# 2170
template< class _Res, class _Class, class _Arg> 
# 2171
struct __result_of_memobj< _Res (_Class::*), _Arg>  { 
# 2174
typedef typename remove_cv< typename remove_reference< _Arg> ::type> ::type _Argval; 
# 2175
typedef _Res (_Class::*_MemPtr); 
# 2180
typedef typename conditional< __or_< is_same< _Argval, _Class> , is_base_of< _Class, _Argval> > ::value, __result_of_memobj_ref< _MemPtr, _Arg> , __result_of_memobj_deref< _MemPtr, _Arg> > ::type::type type; 
# 2181
}; 
# 2183
template< class _MemPtr, class _Arg, class ..._Args> struct __result_of_memfun; 
# 2186
template< class _Res, class _Class, class _Arg, class ..._Args> 
# 2187
struct __result_of_memfun< _Res (_Class::*), _Arg, _Args...>  { 
# 2190
typedef typename remove_cv< typename remove_reference< _Arg> ::type> ::type _Argval; 
# 2191
typedef _Res (_Class::*_MemPtr); 
# 2196
typedef typename conditional< __or_< is_same< _Argval, _Class> , is_base_of< _Class, _Argval> > ::value, __result_of_memfun_ref< _MemPtr, _Arg, _Args...> , __result_of_memfun_deref< _MemPtr, _Arg, _Args...> > ::type::type type; 
# 2197
}; 
# 2204
template< class _Tp, class _Up = typename decay< _Tp> ::type> 
# 2205
struct __inv_unwrap { 
# 2207
using type = _Tp; 
# 2208
}; 
# 2210
template< class _Tp, class _Up> 
# 2211
struct __inv_unwrap< _Tp, reference_wrapper< _Up> >  { 
# 2213
using type = _Up &; 
# 2214
}; 
# 2216
template< bool , bool , class _Functor, class ..._ArgTypes> 
# 2217
struct __result_of_impl { 
# 2219
typedef __failure_type type; 
# 2220
}; 
# 2222
template< class _MemPtr, class _Arg> 
# 2223
struct __result_of_impl< true, false, _MemPtr, _Arg>  : public __result_of_memobj< typename decay< _MemPtr> ::type, typename __inv_unwrap< _Arg> ::type>  { 
# 2226
}; 
# 2228
template< class _MemPtr, class _Arg, class ..._Args> 
# 2229
struct __result_of_impl< false, true, _MemPtr, _Arg, _Args...>  : public __result_of_memfun< typename decay< _MemPtr> ::type, typename __inv_unwrap< _Arg> ::type, _Args...>  { 
# 2232
}; 
# 2235
struct __result_of_other_impl { 
# 2237
template< class _Fn, class ..._Args> static __result_of_success< __decltype((std::declval< _Fn> ()(std::declval< _Args> ()...))), __invoke_other>  _S_test(int); 
# 2242
template< class ...> static __failure_type _S_test(...); 
# 2244
}; 
# 2246
template< class _Functor, class ..._ArgTypes> 
# 2247
struct __result_of_impl< false, false, _Functor, _ArgTypes...>  : private __result_of_other_impl { 
# 2250
typedef __decltype((_S_test< _Functor, _ArgTypes...> (0))) type; 
# 2251
}; 
# 2254
template< class _Functor, class ..._ArgTypes> 
# 2255
struct __invoke_result : public __result_of_impl< is_member_object_pointer< typename remove_reference< _Functor> ::type> ::value, is_member_function_pointer< typename remove_reference< _Functor> ::type> ::value, _Functor, _ArgTypes...> ::type { 
# 2265
}; 
# 2267
template< class _Functor, class ..._ArgTypes> 
# 2268
struct result_of< _Functor (_ArgTypes ...)>  : public __invoke_result< _Functor, _ArgTypes...>  { 
# 2270
}; 
# 2274
template< size_t _Len, size_t _Align = __alignof__(typename __aligned_storage_msa< _Len> ::__type)> using aligned_storage_t = typename aligned_storage< _Len, _Align> ::type; 
# 2278
template< size_t _Len, class ..._Types> using aligned_union_t = typename aligned_union< _Len, _Types...> ::type; 
# 2282
template< class _Tp> using decay_t = typename decay< _Tp> ::type; 
# 2286
template< bool _Cond, class _Tp = void> using enable_if_t = typename enable_if< _Cond, _Tp> ::type; 
# 2290
template< bool _Cond, class _Iftrue, class _Iffalse> using conditional_t = typename conditional< _Cond, _Iftrue, _Iffalse> ::type; 
# 2294
template< class ..._Tp> using common_type_t = typename common_type< _Tp...> ::type; 
# 2298
template< class _Tp> using underlying_type_t = typename underlying_type< _Tp> ::type; 
# 2302
template< class _Tp> using result_of_t = typename result_of< _Tp> ::type; 
# 2307
template< bool _Cond, class _Tp = void> using __enable_if_t = typename enable_if< _Cond, _Tp> ::type; 
# 2311
template< class ...> using __void_t = void; 
# 2316
template< class ...> using void_t = void; 
# 2320
template< class _Default, class _AlwaysVoid, 
# 2321
template< class ...>  class _Op, class ..._Args> 
# 2322
struct __detector { 
# 2324
using value_t = false_type; 
# 2325
using type = _Default; 
# 2326
}; 
# 2329
template< class _Default, template< class ...>  class _Op, class ...
# 2330
_Args> 
# 2331
struct __detector< _Default, __void_t< _Op< _Args...> > , _Op, _Args...>  { 
# 2333
using value_t = true_type; 
# 2334
using type = _Op< _Args...> ; 
# 2335
}; 
# 2338
template< class _Default, template< class ...>  class _Op, class ...
# 2339
_Args> using __detected_or = __detector< _Default, void, _Op, _Args...> ; 
# 2343
template< class _Default, template< class ...>  class _Op, class ...
# 2344
_Args> using __detected_or_t = typename __detector< _Default, void, _Op, _Args...> ::type; 
# 2364 "/usr/include/c++/8/type_traits" 3
template< class _Tp> struct __is_swappable; 
# 2367
template< class _Tp> struct __is_nothrow_swappable; 
# 2370
template< class ..._Elements> class tuple; 
# 2373
template< class > 
# 2374
struct __is_tuple_like_impl : public false_type { 
# 2375
}; 
# 2377
template< class ..._Tps> 
# 2378
struct __is_tuple_like_impl< tuple< _Tps...> >  : public true_type { 
# 2379
}; 
# 2382
template< class _Tp> 
# 2383
struct __is_tuple_like : public __is_tuple_like_impl< typename remove_cv< typename remove_reference< _Tp> ::type> ::type> ::type { 
# 2386
}; 
# 2388
template< class _Tp> inline typename enable_if< __and_< __not_< __is_tuple_like< _Tp> > , is_move_constructible< _Tp> , is_move_assignable< _Tp> > ::value> ::type swap(_Tp &, _Tp &) noexcept(__and_< is_nothrow_move_constructible< _Tp> , is_nothrow_move_assignable< _Tp> > ::value); 
# 2397
template< class _Tp, size_t _Nm> inline typename enable_if< __is_swappable< _Tp> ::value> ::type swap(_Tp (& __a)[_Nm], _Tp (& __b)[_Nm]) noexcept(__is_nothrow_swappable< _Tp> ::value); 
# 2403
namespace __swappable_details { 
# 2404
using std::swap;
# 2406
struct __do_is_swappable_impl { 
# 2408
template< class _Tp, class 
# 2409
 = __decltype((swap(std::declval< _Tp &> (), std::declval< _Tp &> ())))> static true_type 
# 2408
__test(int); 
# 2412
template< class > static false_type __test(...); 
# 2414
}; 
# 2416
struct __do_is_nothrow_swappable_impl { 
# 2418
template< class _Tp> static __bool_constant< noexcept(swap(std::declval< _Tp &> (), std::declval< _Tp &> ()))>  __test(int); 
# 2423
template< class > static false_type __test(...); 
# 2425
}; 
# 2427
}
# 2429
template< class _Tp> 
# 2430
struct __is_swappable_impl : public __swappable_details::__do_is_swappable_impl { 
# 2433
typedef __decltype((__test< _Tp> (0))) type; 
# 2434
}; 
# 2436
template< class _Tp> 
# 2437
struct __is_nothrow_swappable_impl : public __swappable_details::__do_is_nothrow_swappable_impl { 
# 2440
typedef __decltype((__test< _Tp> (0))) type; 
# 2441
}; 
# 2443
template< class _Tp> 
# 2444
struct __is_swappable : public __is_swappable_impl< _Tp> ::type { 
# 2446
}; 
# 2448
template< class _Tp> 
# 2449
struct __is_nothrow_swappable : public __is_nothrow_swappable_impl< _Tp> ::type { 
# 2451
}; 
# 2458
template< class _Tp> 
# 2459
struct is_swappable : public __is_swappable_impl< _Tp> ::type { 
# 2461
}; 
# 2464
template< class _Tp> 
# 2465
struct is_nothrow_swappable : public __is_nothrow_swappable_impl< _Tp> ::type { 
# 2467
}; 
# 2471
template< class _Tp> constexpr bool 
# 2472
is_swappable_v = (is_swappable< _Tp> ::value); 
# 2476
template< class _Tp> constexpr bool 
# 2477
is_nothrow_swappable_v = (is_nothrow_swappable< _Tp> ::value); 
# 2481
namespace __swappable_with_details { 
# 2482
using std::swap;
# 2484
struct __do_is_swappable_with_impl { 
# 2486
template< class _Tp, class _Up, class 
# 2487
 = __decltype((swap(std::declval< _Tp> (), std::declval< _Up> ()))), class 
# 2489
 = __decltype((swap(std::declval< _Up> (), std::declval< _Tp> ())))> static true_type 
# 2486
__test(int); 
# 2492
template< class , class > static false_type __test(...); 
# 2494
}; 
# 2496
struct __do_is_nothrow_swappable_with_impl { 
# 2498
template< class _Tp, class _Up> static __bool_constant< noexcept(swap(std::declval< _Tp> (), std::declval< _Up> ())) && noexcept(swap(std::declval< _Up> (), std::declval< _Tp> ()))>  __test(int); 
# 2505
template< class , class > static false_type __test(...); 
# 2507
}; 
# 2509
}
# 2511
template< class _Tp, class _Up> 
# 2512
struct __is_swappable_with_impl : public __swappable_with_details::__do_is_swappable_with_impl { 
# 2515
typedef __decltype((__test< _Tp, _Up> (0))) type; 
# 2516
}; 
# 2519
template< class _Tp> 
# 2520
struct __is_swappable_with_impl< _Tp &, _Tp &>  : public __swappable_details::__do_is_swappable_impl { 
# 2523
typedef __decltype((__test< _Tp &> (0))) type; 
# 2524
}; 
# 2526
template< class _Tp, class _Up> 
# 2527
struct __is_nothrow_swappable_with_impl : public __swappable_with_details::__do_is_nothrow_swappable_with_impl { 
# 2530
typedef __decltype((__test< _Tp, _Up> (0))) type; 
# 2531
}; 
# 2534
template< class _Tp> 
# 2535
struct __is_nothrow_swappable_with_impl< _Tp &, _Tp &>  : public __swappable_details::__do_is_nothrow_swappable_impl { 
# 2538
typedef __decltype((__test< _Tp &> (0))) type; 
# 2539
}; 
# 2542
template< class _Tp, class _Up> 
# 2543
struct is_swappable_with : public __is_swappable_with_impl< _Tp, _Up> ::type { 
# 2545
}; 
# 2548
template< class _Tp, class _Up> 
# 2549
struct is_nothrow_swappable_with : public __is_nothrow_swappable_with_impl< _Tp, _Up> ::type { 
# 2551
}; 
# 2555
template< class _Tp, class _Up> constexpr bool 
# 2556
is_swappable_with_v = (is_swappable_with< _Tp, _Up> ::value); 
# 2560
template< class _Tp, class _Up> constexpr bool 
# 2561
is_nothrow_swappable_with_v = (is_nothrow_swappable_with< _Tp, _Up> ::value); 
# 2569
template< class _Result, class _Ret, class  = void> 
# 2570
struct __is_invocable_impl : public false_type { }; 
# 2572
template< class _Result, class _Ret> 
# 2573
struct __is_invocable_impl< _Result, _Ret, __void_t< typename _Result::type> >  : public __or_< is_void< _Ret> , is_convertible< typename _Result::type, _Ret> > ::type { 
# 2575
}; 
# 2577
template< class _Fn, class ..._ArgTypes> 
# 2578
struct __is_invocable : public __is_invocable_impl< __invoke_result< _Fn, _ArgTypes...> , void> ::type { 
# 2580
}; 
# 2582
template< class _Fn, class _Tp, class ..._Args> constexpr bool 
# 2583
__call_is_nt(__invoke_memfun_ref) 
# 2584
{ 
# 2585
using _Up = typename __inv_unwrap< _Tp> ::type; 
# 2586
return noexcept((std::declval< typename __inv_unwrap< _Tp> ::type> ().*std::declval< _Fn> ())(std::declval< _Args> ()...)); 
# 2588
} 
# 2590
template< class _Fn, class _Tp, class ..._Args> constexpr bool 
# 2591
__call_is_nt(__invoke_memfun_deref) 
# 2592
{ 
# 2593
return noexcept(((*std::declval< _Tp> ()).*std::declval< _Fn> ())(std::declval< _Args> ()...)); 
# 2595
} 
# 2597
template< class _Fn, class _Tp> constexpr bool 
# 2598
__call_is_nt(__invoke_memobj_ref) 
# 2599
{ 
# 2600
using _Up = typename __inv_unwrap< _Tp> ::type; 
# 2601
return noexcept((std::declval< typename __inv_unwrap< _Tp> ::type> ().*std::declval< _Fn> ())); 
# 2602
} 
# 2604
template< class _Fn, class _Tp> constexpr bool 
# 2605
__call_is_nt(__invoke_memobj_deref) 
# 2606
{ 
# 2607
return noexcept(((*std::declval< _Tp> ()).*std::declval< _Fn> ())); 
# 2608
} 
# 2610
template< class _Fn, class ..._Args> constexpr bool 
# 2611
__call_is_nt(__invoke_other) 
# 2612
{ 
# 2613
return noexcept(std::declval< _Fn> ()(std::declval< _Args> ()...)); 
# 2614
} 
# 2616
template< class _Result, class _Fn, class ..._Args> 
# 2617
struct __call_is_nothrow : public __bool_constant< std::__call_is_nt< _Fn, _Args...> (typename _Result::__invoke_type{})>  { 
# 2621
}; 
# 2623
template< class _Fn, class ..._Args> using __call_is_nothrow_ = __call_is_nothrow< __invoke_result< _Fn, _Args...> , _Fn, _Args...> ; 
# 2628
template< class _Fn, class ..._Args> 
# 2629
struct __is_nothrow_invocable : public __and_< __is_invocable< _Fn, _Args...> , __call_is_nothrow_< _Fn, _Args...> > ::type { 
# 2632
}; 
# 2634
struct __nonesuch { 
# 2635
__nonesuch() = delete;
# 2636
~__nonesuch() = delete;
# 2637
__nonesuch(const __nonesuch &) = delete;
# 2638
void operator=(const __nonesuch &) = delete;
# 2639
}; 
# 2930 "/usr/include/c++/8/type_traits" 3
}
# 57 "/usr/include/c++/8/bits/move.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 72 "/usr/include/c++/8/bits/move.h" 3
template< class _Tp> constexpr _Tp &&
# 74
forward(typename remove_reference< _Tp> ::type &__t) noexcept 
# 75
{ return static_cast< _Tp &&>(__t); } 
# 83
template< class _Tp> constexpr _Tp &&
# 85
forward(typename remove_reference< _Tp> ::type &&__t) noexcept 
# 86
{ 
# 87
static_assert((!std::template is_lvalue_reference< _Tp> ::value), "template argument substituting _Tp is an lvalue reference type");
# 89
return static_cast< _Tp &&>(__t); 
# 90
} 
# 97
template< class _Tp> constexpr typename remove_reference< _Tp> ::type &&
# 99
move(_Tp &&__t) noexcept 
# 100
{ return static_cast< typename remove_reference< _Tp> ::type &&>(__t); } 
# 103
template< class _Tp> 
# 104
struct __move_if_noexcept_cond : public __and_< __not_< is_nothrow_move_constructible< _Tp> > , is_copy_constructible< _Tp> > ::type { 
# 106
}; 
# 116 "/usr/include/c++/8/bits/move.h" 3
template< class _Tp> constexpr typename conditional< __move_if_noexcept_cond< _Tp> ::value, const _Tp &, _Tp &&> ::type 
# 119
move_if_noexcept(_Tp &__x) noexcept 
# 120
{ return std::move(__x); } 
# 136 "/usr/include/c++/8/bits/move.h" 3
template< class _Tp> inline _Tp *
# 138
addressof(_Tp &__r) noexcept 
# 139
{ return std::__addressof(__r); } 
# 143
template < typename _Tp >
    const _Tp * addressof ( const _Tp && ) = delete;
# 147
template< class _Tp, class _Up = _Tp> inline _Tp 
# 149
__exchange(_Tp &__obj, _Up &&__new_val) 
# 150
{ 
# 151
_Tp __old_val = std::move(__obj); 
# 152
__obj = std::forward< _Up> (__new_val); 
# 153
return __old_val; 
# 154
} 
# 176 "/usr/include/c++/8/bits/move.h" 3
template< class _Tp> inline typename enable_if< __and_< __not_< __is_tuple_like< _Tp> > , is_move_constructible< _Tp> , is_move_assignable< _Tp> > ::value> ::type 
# 182
swap(_Tp &__a, _Tp &__b) noexcept(__and_< is_nothrow_move_constructible< _Tp> , is_nothrow_move_assignable< _Tp> > ::value) 
# 189
{ 
# 193
_Tp __tmp = std::move(__a); 
# 194
__a = std::move(__b); 
# 195
__b = std::move(__tmp); 
# 196
} 
# 201
template< class _Tp, size_t _Nm> inline typename enable_if< __is_swappable< _Tp> ::value> ::type 
# 205
swap(_Tp (&__a)[_Nm], _Tp (&__b)[_Nm]) noexcept(__is_nothrow_swappable< _Tp> ::value) 
# 211
{ 
# 212
for (size_t __n = (0); __n < _Nm; ++__n) { 
# 213
swap(__a[__n], __b[__n]); }  
# 214
} 
# 218
}
# 65 "/usr/include/c++/8/bits/stl_pair.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 76 "/usr/include/c++/8/bits/stl_pair.h" 3
struct piecewise_construct_t { explicit piecewise_construct_t() = default;}; 
# 79
constexpr piecewise_construct_t piecewise_construct = piecewise_construct_t(); 
# 83
template< class ...> class tuple; 
# 86
template< size_t ...> struct _Index_tuple; 
# 94
template< bool , class _T1, class _T2> 
# 95
struct _PCC { 
# 97
template< class _U1, class _U2> static constexpr bool 
# 98
_ConstructiblePair() 
# 99
{ 
# 100
return __and_< is_constructible< _T1, const _U1 &> , is_constructible< _T2, const _U2 &> > ::value; 
# 102
} 
# 104
template< class _U1, class _U2> static constexpr bool 
# 105
_ImplicitlyConvertiblePair() 
# 106
{ 
# 107
return __and_< is_convertible< const _U1 &, _T1> , is_convertible< const _U2 &, _T2> > ::value; 
# 109
} 
# 111
template< class _U1, class _U2> static constexpr bool 
# 112
_MoveConstructiblePair() 
# 113
{ 
# 114
return __and_< is_constructible< _T1, _U1 &&> , is_constructible< _T2, _U2 &&> > ::value; 
# 116
} 
# 118
template< class _U1, class _U2> static constexpr bool 
# 119
_ImplicitlyMoveConvertiblePair() 
# 120
{ 
# 121
return __and_< is_convertible< _U1 &&, _T1> , is_convertible< _U2 &&, _T2> > ::value; 
# 123
} 
# 125
template< bool __implicit, class _U1, class _U2> static constexpr bool 
# 126
_CopyMovePair() 
# 127
{ 
# 128
using __do_converts = __and_< is_convertible< const _U1 &, _T1> , is_convertible< _U2 &&, _T2> > ; 
# 130
using __converts = typename conditional< __implicit, __and_< is_convertible< const _U1 &, _T1> , is_convertible< _U2 &&, _T2> > , __not_< __and_< is_convertible< const _U1 &, _T1> , is_convertible< _U2 &&, _T2> > > > ::type; 
# 133
return __and_< is_constructible< _T1, const _U1 &> , is_constructible< _T2, _U2 &&> , typename conditional< __implicit, __and_< is_convertible< const _U1 &, _T1> , is_convertible< _U2 &&, _T2> > , __not_< __and_< is_convertible< const _U1 &, _T1> , is_convertible< _U2 &&, _T2> > > > ::type> ::value; 
# 137
} 
# 139
template< bool __implicit, class _U1, class _U2> static constexpr bool 
# 140
_MoveCopyPair() 
# 141
{ 
# 142
using __do_converts = __and_< is_convertible< _U1 &&, _T1> , is_convertible< const _U2 &, _T2> > ; 
# 144
using __converts = typename conditional< __implicit, __and_< is_convertible< _U1 &&, _T1> , is_convertible< const _U2 &, _T2> > , __not_< __and_< is_convertible< _U1 &&, _T1> , is_convertible< const _U2 &, _T2> > > > ::type; 
# 147
return __and_< is_constructible< _T1, _U1 &&> , is_constructible< _T2, const _U2 &&> , typename conditional< __implicit, __and_< is_convertible< _U1 &&, _T1> , is_convertible< const _U2 &, _T2> > , __not_< __and_< is_convertible< _U1 &&, _T1> , is_convertible< const _U2 &, _T2> > > > ::type> ::value; 
# 151
} 
# 152
}; 
# 154
template< class _T1, class _T2> 
# 155
struct _PCC< false, _T1, _T2>  { 
# 157
template< class _U1, class _U2> static constexpr bool 
# 158
_ConstructiblePair() 
# 159
{ 
# 160
return false; 
# 161
} 
# 163
template< class _U1, class _U2> static constexpr bool 
# 164
_ImplicitlyConvertiblePair() 
# 165
{ 
# 166
return false; 
# 167
} 
# 169
template< class _U1, class _U2> static constexpr bool 
# 170
_MoveConstructiblePair() 
# 171
{ 
# 172
return false; 
# 173
} 
# 175
template< class _U1, class _U2> static constexpr bool 
# 176
_ImplicitlyMoveConvertiblePair() 
# 177
{ 
# 178
return false; 
# 179
} 
# 180
}; 
# 185
struct __nonesuch_no_braces : public __nonesuch { 
# 186
explicit __nonesuch_no_braces(const __nonesuch &) = delete;
# 187
}; 
# 190
template< class _U1, class _U2> class __pair_base { 
# 193
template< class _T1, class _T2> friend struct pair; 
# 194
__pair_base() = default;
# 195
~__pair_base() = default;
# 196
__pair_base(const __pair_base &) = default;
# 197
__pair_base &operator=(const __pair_base &) = delete;
# 199
}; 
# 207
template< class _T1, class _T2> 
# 208
struct pair : private __pair_base< _T1, _T2>  { 
# 211
typedef _T1 first_type; 
# 212
typedef _T2 second_type; 
# 214
_T1 first; 
# 215
_T2 second; 
# 222
template< class _U1 = _T1, class 
# 223
_U2 = _T2, typename enable_if< __and_< __is_implicitly_default_constructible< _U1> , __is_implicitly_default_constructible< _U2> > ::value, bool> ::type 
# 227
 = true> constexpr 
# 229
pair() : first(), second() 
# 230
{ } 
# 233
template< class _U1 = _T1, class 
# 234
_U2 = _T2, typename enable_if< __and_< is_default_constructible< _U1> , is_default_constructible< _U2> , __not_< __and_< __is_implicitly_default_constructible< _U1> , __is_implicitly_default_constructible< _U2> > > > ::value, bool> ::type 
# 241
 = false> constexpr explicit 
# 242
pair() : first(), second() 
# 243
{ } 
# 252 "/usr/include/c++/8/bits/stl_pair.h" 3
using _PCCP = _PCC< true, _T1, _T2> ; 
# 254
template< class _U1 = _T1, class _U2 = _T2, typename enable_if< _PCC< true, _T1, _T2> ::template _ConstructiblePair< _U1, _U2> () && _PCC< true, _T1, _T2> ::template _ImplicitlyConvertiblePair< _U1, _U2> (), bool> ::type 
# 259
 = true> constexpr 
# 260
pair(const _T1 &__a, const _T2 &__b) : first(__a), second(__b) 
# 261
{ } 
# 263
template< class _U1 = _T1, class _U2 = _T2, typename enable_if< _PCC< true, _T1, _T2> ::template _ConstructiblePair< _U1, _U2> () && (!_PCC< true, _T1, _T2> ::template _ImplicitlyConvertiblePair< _U1, _U2> ()), bool> ::type 
# 268
 = false> constexpr explicit 
# 269
pair(const _T1 &__a, const _T2 &__b) : first(__a), second(__b) 
# 270
{ } 
# 280 "/usr/include/c++/8/bits/stl_pair.h" 3
template< class _U1, class _U2> using _PCCFP = _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ; 
# 285
template< class _U1, class _U2, typename enable_if< _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ConstructiblePair< _U1, _U2> () && _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ImplicitlyConvertiblePair< _U1, _U2> (), bool> ::type 
# 290
 = true> constexpr 
# 291
pair(const std::pair< _U1, _U2>  &__p) : first((__p.first)), second((__p.second)) 
# 292
{ } 
# 294
template< class _U1, class _U2, typename enable_if< _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ConstructiblePair< _U1, _U2> () && (!_PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ImplicitlyConvertiblePair< _U1, _U2> ()), bool> ::type 
# 299
 = false> constexpr explicit 
# 300
pair(const std::pair< _U1, _U2>  &__p) : first((__p.first)), second((__p.second)) 
# 301
{ } 
# 303
constexpr pair(const pair &) = default;
# 304
constexpr pair(pair &&) = default;
# 307
template< class _U1, typename enable_if< _PCC< true, _T1, _T2> ::template _MoveCopyPair< true, _U1, _T2> (), bool> ::type 
# 310
 = true> constexpr 
# 311
pair(_U1 &&__x, const _T2 &__y) : first(std::forward< _U1> (__x)), second(__y) 
# 312
{ } 
# 314
template< class _U1, typename enable_if< _PCC< true, _T1, _T2> ::template _MoveCopyPair< false, _U1, _T2> (), bool> ::type 
# 317
 = false> constexpr explicit 
# 318
pair(_U1 &&__x, const _T2 &__y) : first(std::forward< _U1> (__x)), second(__y) 
# 319
{ } 
# 321
template< class _U2, typename enable_if< _PCC< true, _T1, _T2> ::template _CopyMovePair< true, _T1, _U2> (), bool> ::type 
# 324
 = true> constexpr 
# 325
pair(const _T1 &__x, _U2 &&__y) : first(__x), second(std::forward< _U2> (__y)) 
# 326
{ } 
# 328
template< class _U2, typename enable_if< _PCC< true, _T1, _T2> ::template _CopyMovePair< false, _T1, _U2> (), bool> ::type 
# 331
 = false> explicit 
# 332
pair(const _T1 &__x, _U2 &&__y) : first(__x), second(std::forward< _U2> (__y)) 
# 333
{ } 
# 335
template< class _U1, class _U2, typename enable_if< _PCC< true, _T1, _T2> ::template _MoveConstructiblePair< _U1, _U2> () && _PCC< true, _T1, _T2> ::template _ImplicitlyMoveConvertiblePair< _U1, _U2> (), bool> ::type 
# 340
 = true> constexpr 
# 341
pair(_U1 &&__x, _U2 &&__y) : first(std::forward< _U1> (__x)), second(std::forward< _U2> (__y)) 
# 342
{ } 
# 344
template< class _U1, class _U2, typename enable_if< _PCC< true, _T1, _T2> ::template _MoveConstructiblePair< _U1, _U2> () && (!_PCC< true, _T1, _T2> ::template _ImplicitlyMoveConvertiblePair< _U1, _U2> ()), bool> ::type 
# 349
 = false> constexpr explicit 
# 350
pair(_U1 &&__x, _U2 &&__y) : first(std::forward< _U1> (__x)), second(std::forward< _U2> (__y)) 
# 351
{ } 
# 354
template< class _U1, class _U2, typename enable_if< _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _MoveConstructiblePair< _U1, _U2> () && _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ImplicitlyMoveConvertiblePair< _U1, _U2> (), bool> ::type 
# 359
 = true> constexpr 
# 360
pair(std::pair< _U1, _U2>  &&__p) : first(std::forward< _U1> ((__p.first))), second(std::forward< _U2> ((__p.second))) 
# 362
{ } 
# 364
template< class _U1, class _U2, typename enable_if< _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _MoveConstructiblePair< _U1, _U2> () && (!_PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ImplicitlyMoveConvertiblePair< _U1, _U2> ()), bool> ::type 
# 369
 = false> constexpr explicit 
# 370
pair(std::pair< _U1, _U2>  &&__p) : first(std::forward< _U1> ((__p.first))), second(std::forward< _U2> ((__p.second))) 
# 372
{ } 
# 374
template< class ..._Args1, class ..._Args2> pair(std::piecewise_construct_t, tuple< _Args1...> , tuple< _Args2...> ); 
# 378
pair &operator=(typename conditional< __and_< is_copy_assignable< _T1> , is_copy_assignable< _T2> > ::value, const pair &, const std::__nonesuch_no_braces &> ::type 
# 381
__p) 
# 382
{ 
# 383
(first) = (__p.first); 
# 384
(second) = (__p.second); 
# 385
return *this; 
# 386
} 
# 389
pair &operator=(typename conditional< __and_< is_move_assignable< _T1> , is_move_assignable< _T2> > ::value, pair &&, std::__nonesuch_no_braces &&> ::type 
# 392
__p) noexcept(__and_< is_nothrow_move_assignable< _T1> , is_nothrow_move_assignable< _T2> > ::value) 
# 395
{ 
# 396
(first) = std::forward< first_type> ((__p.first)); 
# 397
(second) = std::forward< second_type> ((__p.second)); 
# 398
return *this; 
# 399
} 
# 401
template< class _U1, class _U2> typename enable_if< __and_< is_assignable< _T1 &, const _U1 &> , is_assignable< _T2 &, const _U2 &> > ::value, pair &> ::type 
# 405
operator=(const std::pair< _U1, _U2>  &__p) 
# 406
{ 
# 407
(first) = (__p.first); 
# 408
(second) = (__p.second); 
# 409
return *this; 
# 410
} 
# 412
template< class _U1, class _U2> typename enable_if< __and_< is_assignable< _T1 &, _U1 &&> , is_assignable< _T2 &, _U2 &&> > ::value, pair &> ::type 
# 416
operator=(std::pair< _U1, _U2>  &&__p) 
# 417
{ 
# 418
(first) = std::forward< _U1> ((__p.first)); 
# 419
(second) = std::forward< _U2> ((__p.second)); 
# 420
return *this; 
# 421
} 
# 424
void swap(pair &__p) noexcept(__and_< __is_nothrow_swappable< _T1> , __is_nothrow_swappable< _T2> > ::value) 
# 427
{ 
# 428
using std::swap;
# 429
swap(first, __p.first); 
# 430
swap(second, __p.second); 
# 431
} 
# 434
private: template< class ..._Args1, std::size_t ..._Indexes1, class ...
# 435
_Args2, std::size_t ..._Indexes2> 
# 434
pair(tuple< _Args1...>  &, tuple< _Args2...>  &, _Index_tuple< _Indexes1...> , _Index_tuple< _Indexes2...> ); 
# 439
}; 
# 446
template< class _T1, class _T2> constexpr bool 
# 448
operator==(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 449
{ return ((__x.first) == (__y.first)) && ((__x.second) == (__y.second)); } 
# 452
template< class _T1, class _T2> constexpr bool 
# 454
operator<(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 455
{ return ((__x.first) < (__y.first)) || ((!((__y.first) < (__x.first))) && ((__x.second) < (__y.second))); 
# 456
} 
# 459
template< class _T1, class _T2> constexpr bool 
# 461
operator!=(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 462
{ return !(__x == __y); } 
# 465
template< class _T1, class _T2> constexpr bool 
# 467
operator>(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 468
{ return __y < __x; } 
# 471
template< class _T1, class _T2> constexpr bool 
# 473
operator<=(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 474
{ return !(__y < __x); } 
# 477
template< class _T1, class _T2> constexpr bool 
# 479
operator>=(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 480
{ return !(__x < __y); } 
# 486
template< class _T1, class _T2> inline typename enable_if< __and_< __is_swappable< _T1> , __is_swappable< _T2> > ::value> ::type 
# 495
swap(pair< _T1, _T2>  &__x, pair< _T1, _T2>  &__y) noexcept(noexcept(__x.swap(__y))) 
# 497
{ __x.swap(__y); } 
# 500
template < typename _T1, typename _T2 >
    typename enable_if < ! __and_ < __is_swappable < _T1 >,
          __is_swappable < _T2 > > :: value > :: type
    swap ( pair < _T1, _T2 > &, pair < _T1, _T2 > & ) = delete;
# 521 "/usr/include/c++/8/bits/stl_pair.h" 3
template< class _T1, class _T2> constexpr pair< typename __decay_and_strip< _T1> ::__type, typename __decay_and_strip< _T2> ::__type>  
# 524
make_pair(_T1 &&__x, _T2 &&__y) 
# 525
{ 
# 526
typedef typename __decay_and_strip< _T1> ::__type __ds_type1; 
# 527
typedef typename __decay_and_strip< _T2> ::__type __ds_type2; 
# 528
typedef pair< typename __decay_and_strip< _T1> ::__type, typename __decay_and_strip< _T2> ::__type>  __pair_type; 
# 529
return __pair_type(std::forward< _T1> (__x), std::forward< _T2> (__y)); 
# 530
} 
# 541 "/usr/include/c++/8/bits/stl_pair.h" 3
}
# 39 "/usr/include/c++/8/initializer_list" 3
#pragma GCC visibility push ( default )
# 43
namespace std { 
# 46
template< class _E> 
# 47
class initializer_list { 
# 50
public: typedef _E value_type; 
# 51
typedef const _E &reference; 
# 52
typedef const _E &const_reference; 
# 53
typedef size_t size_type; 
# 54
typedef const _E *iterator; 
# 55
typedef const _E *const_iterator; 
# 58
private: iterator _M_array; 
# 59
size_type _M_len; 
# 62
constexpr initializer_list(const_iterator __a, size_type __l) : _M_array(__a), _M_len(__l) 
# 63
{ } 
# 66
public: constexpr initializer_list() noexcept : _M_array((0)), _M_len((0)) 
# 67
{ } 
# 71
constexpr size_type size() const noexcept { return _M_len; } 
# 75
constexpr const_iterator begin() const noexcept { return _M_array; } 
# 79
constexpr const_iterator end() const noexcept { return begin() + size(); } 
# 80
}; 
# 87
template< class _Tp> constexpr const _Tp *
# 89
begin(initializer_list< _Tp>  __ils) noexcept 
# 90
{ return __ils.begin(); } 
# 97
template< class _Tp> constexpr const _Tp *
# 99
end(initializer_list< _Tp>  __ils) noexcept 
# 100
{ return __ils.end(); } 
# 101
}
# 103
#pragma GCC visibility pop
# 78 "/usr/include/c++/8/utility" 3
namespace std __attribute((__visibility__("default"))) { 
# 83
template< class _Tp> struct tuple_size; 
# 90
template< class _Tp, class 
# 91
_Up = typename remove_cv< _Tp> ::type, class 
# 92
 = typename enable_if< is_same< _Tp, _Up> ::value> ::type, size_t 
# 93
 = tuple_size< _Tp> ::value> using __enable_if_has_tuple_size = _Tp; 
# 96
template< class _Tp> 
# 97
struct tuple_size< const __enable_if_has_tuple_size< _Tp> >  : public std::tuple_size< _Tp>  { 
# 98
}; 
# 100
template< class _Tp> 
# 101
struct tuple_size< volatile __enable_if_has_tuple_size< _Tp> >  : public std::tuple_size< _Tp>  { 
# 102
}; 
# 104
template< class _Tp> 
# 105
struct tuple_size< const volatile __enable_if_has_tuple_size< _Tp> >  : public std::tuple_size< _Tp>  { 
# 106
}; 
# 109
template< size_t __i, class _Tp> struct tuple_element; 
# 113
template< size_t __i, class _Tp> using __tuple_element_t = typename tuple_element< __i, _Tp> ::type; 
# 116
template< size_t __i, class _Tp> 
# 117
struct tuple_element< __i, const _Tp>  { 
# 119
typedef typename add_const< __tuple_element_t< __i, _Tp> > ::type type; 
# 120
}; 
# 122
template< size_t __i, class _Tp> 
# 123
struct tuple_element< __i, volatile _Tp>  { 
# 125
typedef typename add_volatile< __tuple_element_t< __i, _Tp> > ::type type; 
# 126
}; 
# 128
template< size_t __i, class _Tp> 
# 129
struct tuple_element< __i, const volatile _Tp>  { 
# 131
typedef typename add_cv< __tuple_element_t< __i, _Tp> > ::type type; 
# 132
}; 
# 137
template< size_t __i, class _Tp> using tuple_element_t = typename tuple_element< __i, _Tp> ::type; 
# 144
template< class _T1, class _T2> 
# 145
struct __is_tuple_like_impl< pair< _T1, _T2> >  : public true_type { 
# 146
}; 
# 149
template< class _Tp1, class _Tp2> 
# 150
struct tuple_size< pair< _Tp1, _Tp2> >  : public integral_constant< unsigned long, 2UL>  { 
# 151
}; 
# 154
template< class _Tp1, class _Tp2> 
# 155
struct tuple_element< 0, pair< _Tp1, _Tp2> >  { 
# 156
typedef _Tp1 type; }; 
# 159
template< class _Tp1, class _Tp2> 
# 160
struct tuple_element< 1, pair< _Tp1, _Tp2> >  { 
# 161
typedef _Tp2 type; }; 
# 163
template< size_t _Int> struct __pair_get; 
# 167
template<> struct __pair_get< 0UL>  { 
# 169
template< class _Tp1, class _Tp2> static constexpr _Tp1 &
# 171
__get(pair< _Tp1, _Tp2>  &__pair) noexcept 
# 172
{ return __pair.first; } 
# 174
template< class _Tp1, class _Tp2> static constexpr _Tp1 &&
# 176
__move_get(pair< _Tp1, _Tp2>  &&__pair) noexcept 
# 177
{ return std::forward< _Tp1> ((__pair.first)); } 
# 179
template< class _Tp1, class _Tp2> static constexpr const _Tp1 &
# 181
__const_get(const pair< _Tp1, _Tp2>  &__pair) noexcept 
# 182
{ return __pair.first; } 
# 184
template< class _Tp1, class _Tp2> static constexpr const _Tp1 &&
# 186
__const_move_get(const pair< _Tp1, _Tp2>  &&__pair) noexcept 
# 187
{ return std::forward< const _Tp1> ((__pair.first)); } 
# 188
}; 
# 191
template<> struct __pair_get< 1UL>  { 
# 193
template< class _Tp1, class _Tp2> static constexpr _Tp2 &
# 195
__get(pair< _Tp1, _Tp2>  &__pair) noexcept 
# 196
{ return __pair.second; } 
# 198
template< class _Tp1, class _Tp2> static constexpr _Tp2 &&
# 200
__move_get(pair< _Tp1, _Tp2>  &&__pair) noexcept 
# 201
{ return std::forward< _Tp2> ((__pair.second)); } 
# 203
template< class _Tp1, class _Tp2> static constexpr const _Tp2 &
# 205
__const_get(const pair< _Tp1, _Tp2>  &__pair) noexcept 
# 206
{ return __pair.second; } 
# 208
template< class _Tp1, class _Tp2> static constexpr const _Tp2 &&
# 210
__const_move_get(const pair< _Tp1, _Tp2>  &&__pair) noexcept 
# 211
{ return std::forward< const _Tp2> ((__pair.second)); } 
# 212
}; 
# 214
template< size_t _Int, class _Tp1, class _Tp2> constexpr typename tuple_element< _Int, pair< _Tp1, _Tp2> > ::type &
# 216
get(pair< _Tp1, _Tp2>  &__in) noexcept 
# 217
{ return __pair_get< _Int> ::__get(__in); } 
# 219
template< size_t _Int, class _Tp1, class _Tp2> constexpr typename tuple_element< _Int, pair< _Tp1, _Tp2> > ::type &&
# 221
get(pair< _Tp1, _Tp2>  &&__in) noexcept 
# 222
{ return __pair_get< _Int> ::__move_get(std::move(__in)); } 
# 224
template< size_t _Int, class _Tp1, class _Tp2> constexpr const typename tuple_element< _Int, pair< _Tp1, _Tp2> > ::type &
# 226
get(const pair< _Tp1, _Tp2>  &__in) noexcept 
# 227
{ return __pair_get< _Int> ::__const_get(__in); } 
# 229
template< size_t _Int, class _Tp1, class _Tp2> constexpr const typename tuple_element< _Int, pair< _Tp1, _Tp2> > ::type &&
# 231
get(const pair< _Tp1, _Tp2>  &&__in) noexcept 
# 232
{ return __pair_get< _Int> ::__const_move_get(std::move(__in)); } 
# 238
template< class _Tp, class _Up> constexpr _Tp &
# 240
get(pair< _Tp, _Up>  &__p) noexcept 
# 241
{ return __p.first; } 
# 243
template< class _Tp, class _Up> constexpr const _Tp &
# 245
get(const pair< _Tp, _Up>  &__p) noexcept 
# 246
{ return __p.first; } 
# 248
template< class _Tp, class _Up> constexpr _Tp &&
# 250
get(pair< _Tp, _Up>  &&__p) noexcept 
# 251
{ return std::move((__p.first)); } 
# 253
template< class _Tp, class _Up> constexpr const _Tp &&
# 255
get(const pair< _Tp, _Up>  &&__p) noexcept 
# 256
{ return std::move((__p.first)); } 
# 258
template< class _Tp, class _Up> constexpr _Tp &
# 260
get(pair< _Up, _Tp>  &__p) noexcept 
# 261
{ return __p.second; } 
# 263
template< class _Tp, class _Up> constexpr const _Tp &
# 265
get(const pair< _Up, _Tp>  &__p) noexcept 
# 266
{ return __p.second; } 
# 268
template< class _Tp, class _Up> constexpr _Tp &&
# 270
get(pair< _Up, _Tp>  &&__p) noexcept 
# 271
{ return std::move((__p.second)); } 
# 273
template< class _Tp, class _Up> constexpr const _Tp &&
# 275
get(const pair< _Up, _Tp>  &&__p) noexcept 
# 276
{ return std::move((__p.second)); } 
# 281
template< class _Tp, class _Up = _Tp> inline _Tp 
# 283
exchange(_Tp &__obj, _Up &&__new_val) 
# 284
{ return std::__exchange(__obj, std::forward< _Up> (__new_val)); } 
# 289
template< size_t ..._Indexes> struct _Index_tuple { }; 
# 298 "/usr/include/c++/8/utility" 3
template< size_t _Num> 
# 299
struct _Build_index_tuple { 
# 307
using __type = _Index_tuple< __integer_pack(_Num)...> ; 
# 309
}; 
# 316
template< class _Tp, _Tp ..._Idx> 
# 317
struct integer_sequence { 
# 319
typedef _Tp value_type; 
# 320
static constexpr size_t size() noexcept { return sizeof...(_Idx); } 
# 321
}; 
# 324
template< class _Tp, _Tp _Num> using make_integer_sequence = integer_sequence< _Tp, __integer_pack(_Num)...> ; 
# 335
template< size_t ..._Idx> using index_sequence = integer_sequence< unsigned long, _Idx...> ; 
# 339
template< size_t _Num> using make_index_sequence = make_integer_sequence< unsigned long, _Num> ; 
# 343
template< class ..._Types> using index_sequence_for = make_index_sequence< sizeof...(_Types)> ; 
# 394 "/usr/include/c++/8/utility" 3
}
# 205 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 206
cudaLaunchKernel(const T *
# 207
func, dim3 
# 208
gridDim, dim3 
# 209
blockDim, void **
# 210
args, size_t 
# 211
sharedMem = 0, cudaStream_t 
# 212
stream = 0) 
# 214
{ 
# 215
return ::cudaLaunchKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
# 216
} 
# 276 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class ...ExpTypes, class ...ActTypes> static inline cudaError_t 
# 277
cudaLaunchKernelEx(const cudaLaunchConfig_t *
# 278
config, void (*
# 279
kernel)(ExpTypes ...), ActTypes &&...
# 280
args) 
# 282
{ 
# 283
return [&](ExpTypes ...coercedArgs) { 
# 284
void *pArgs[] = {(&coercedArgs)...}; 
# 285
return ::cudaLaunchKernelExC(config, (const void *)(kernel), pArgs); 
# 286
} (std::forward< ActTypes> (args)...); 
# 287
} 
# 339 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 340
cudaLaunchCooperativeKernel(const T *
# 341
func, dim3 
# 342
gridDim, dim3 
# 343
blockDim, void **
# 344
args, size_t 
# 345
sharedMem = 0, cudaStream_t 
# 346
stream = 0) 
# 348
{ 
# 349
return ::cudaLaunchCooperativeKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
# 350
} 
# 383 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
static inline cudaError_t cudaEventCreate(cudaEvent_t *
# 384
event, unsigned 
# 385
flags) 
# 387
{ 
# 388
return ::cudaEventCreateWithFlags(event, flags); 
# 389
} 
# 448 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
static inline cudaError_t cudaMallocHost(void **
# 449
ptr, size_t 
# 450
size, unsigned 
# 451
flags) 
# 453
{ 
# 454
return ::cudaHostAlloc(ptr, size, flags); 
# 455
} 
# 457
template< class T> static inline cudaError_t 
# 458
cudaHostAlloc(T **
# 459
ptr, size_t 
# 460
size, unsigned 
# 461
flags) 
# 463
{ 
# 464
return ::cudaHostAlloc((void **)((void *)ptr), size, flags); 
# 465
} 
# 467
template< class T> static inline cudaError_t 
# 468
cudaHostGetDevicePointer(T **
# 469
pDevice, void *
# 470
pHost, unsigned 
# 471
flags) 
# 473
{ 
# 474
return ::cudaHostGetDevicePointer((void **)((void *)pDevice), pHost, flags); 
# 475
} 
# 577 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 578
cudaMallocManaged(T **
# 579
devPtr, size_t 
# 580
size, unsigned 
# 581
flags = 1) 
# 583
{ 
# 584
return ::cudaMallocManaged((void **)((void *)devPtr), size, flags); 
# 585
} 
# 667 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 668
cudaStreamAttachMemAsync(cudaStream_t 
# 669
stream, T *
# 670
devPtr, size_t 
# 671
length = 0, unsigned 
# 672
flags = 4) 
# 674
{ 
# 675
return ::cudaStreamAttachMemAsync(stream, (void *)devPtr, length, flags); 
# 676
} 
# 678
template< class T> inline cudaError_t 
# 679
cudaMalloc(T **
# 680
devPtr, size_t 
# 681
size) 
# 683
{ 
# 684
return ::cudaMalloc((void **)((void *)devPtr), size); 
# 685
} 
# 687
template< class T> static inline cudaError_t 
# 688
cudaMallocHost(T **
# 689
ptr, size_t 
# 690
size, unsigned 
# 691
flags = 0) 
# 693
{ 
# 694
return cudaMallocHost((void **)((void *)ptr), size, flags); 
# 695
} 
# 697
template< class T> static inline cudaError_t 
# 698
cudaMallocPitch(T **
# 699
devPtr, size_t *
# 700
pitch, size_t 
# 701
width, size_t 
# 702
height) 
# 704
{ 
# 705
return ::cudaMallocPitch((void **)((void *)devPtr), pitch, width, height); 
# 706
} 
# 717 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
static inline cudaError_t cudaMallocAsync(void **
# 718
ptr, size_t 
# 719
size, cudaMemPool_t 
# 720
memPool, cudaStream_t 
# 721
stream) 
# 723
{ 
# 724
return ::cudaMallocFromPoolAsync(ptr, size, memPool, stream); 
# 725
} 
# 727
template< class T> static inline cudaError_t 
# 728
cudaMallocAsync(T **
# 729
ptr, size_t 
# 730
size, cudaMemPool_t 
# 731
memPool, cudaStream_t 
# 732
stream) 
# 734
{ 
# 735
return ::cudaMallocFromPoolAsync((void **)((void *)ptr), size, memPool, stream); 
# 736
} 
# 738
template< class T> static inline cudaError_t 
# 739
cudaMallocAsync(T **
# 740
ptr, size_t 
# 741
size, cudaStream_t 
# 742
stream) 
# 744
{ 
# 745
return ::cudaMallocAsync((void **)((void *)ptr), size, stream); 
# 746
} 
# 748
template< class T> static inline cudaError_t 
# 749
cudaMallocFromPoolAsync(T **
# 750
ptr, size_t 
# 751
size, cudaMemPool_t 
# 752
memPool, cudaStream_t 
# 753
stream) 
# 755
{ 
# 756
return ::cudaMallocFromPoolAsync((void **)((void *)ptr), size, memPool, stream); 
# 757
} 
# 796 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 797
cudaMemcpyToSymbol(const T &
# 798
symbol, const void *
# 799
src, size_t 
# 800
count, size_t 
# 801
offset = 0, cudaMemcpyKind 
# 802
kind = cudaMemcpyHostToDevice) 
# 804
{ 
# 805
return ::cudaMemcpyToSymbol((const void *)(&symbol), src, count, offset, kind); 
# 806
} 
# 850 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 851
cudaMemcpyToSymbolAsync(const T &
# 852
symbol, const void *
# 853
src, size_t 
# 854
count, size_t 
# 855
offset = 0, cudaMemcpyKind 
# 856
kind = cudaMemcpyHostToDevice, cudaStream_t 
# 857
stream = 0) 
# 859
{ 
# 860
return ::cudaMemcpyToSymbolAsync((const void *)(&symbol), src, count, offset, kind, stream); 
# 861
} 
# 898 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 899
cudaMemcpyFromSymbol(void *
# 900
dst, const T &
# 901
symbol, size_t 
# 902
count, size_t 
# 903
offset = 0, cudaMemcpyKind 
# 904
kind = cudaMemcpyDeviceToHost) 
# 906
{ 
# 907
return ::cudaMemcpyFromSymbol(dst, (const void *)(&symbol), count, offset, kind); 
# 908
} 
# 952 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 953
cudaMemcpyFromSymbolAsync(void *
# 954
dst, const T &
# 955
symbol, size_t 
# 956
count, size_t 
# 957
offset = 0, cudaMemcpyKind 
# 958
kind = cudaMemcpyDeviceToHost, cudaStream_t 
# 959
stream = 0) 
# 961
{ 
# 962
return ::cudaMemcpyFromSymbolAsync(dst, (const void *)(&symbol), count, offset, kind, stream); 
# 963
} 
# 1021 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1022
cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t *
# 1023
pGraphNode, cudaGraph_t 
# 1024
graph, const cudaGraphNode_t *
# 1025
pDependencies, size_t 
# 1026
numDependencies, const T &
# 1027
symbol, const void *
# 1028
src, size_t 
# 1029
count, size_t 
# 1030
offset, cudaMemcpyKind 
# 1031
kind) 
# 1032
{ 
# 1033
return ::cudaGraphAddMemcpyNodeToSymbol(pGraphNode, graph, pDependencies, numDependencies, (const void *)(&symbol), src, count, offset, kind); 
# 1034
} 
# 1092 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1093
cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t *
# 1094
pGraphNode, cudaGraph_t 
# 1095
graph, const cudaGraphNode_t *
# 1096
pDependencies, size_t 
# 1097
numDependencies, void *
# 1098
dst, const T &
# 1099
symbol, size_t 
# 1100
count, size_t 
# 1101
offset, cudaMemcpyKind 
# 1102
kind) 
# 1103
{ 
# 1104
return ::cudaGraphAddMemcpyNodeFromSymbol(pGraphNode, graph, pDependencies, numDependencies, dst, (const void *)(&symbol), count, offset, kind); 
# 1105
} 
# 1143 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1144
cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t 
# 1145
node, const T &
# 1146
symbol, const void *
# 1147
src, size_t 
# 1148
count, size_t 
# 1149
offset, cudaMemcpyKind 
# 1150
kind) 
# 1151
{ 
# 1152
return ::cudaGraphMemcpyNodeSetParamsToSymbol(node, (const void *)(&symbol), src, count, offset, kind); 
# 1153
} 
# 1191 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1192
cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t 
# 1193
node, void *
# 1194
dst, const T &
# 1195
symbol, size_t 
# 1196
count, size_t 
# 1197
offset, cudaMemcpyKind 
# 1198
kind) 
# 1199
{ 
# 1200
return ::cudaGraphMemcpyNodeSetParamsFromSymbol(node, dst, (const void *)(&symbol), count, offset, kind); 
# 1201
} 
# 1249 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1250
cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t 
# 1251
hGraphExec, cudaGraphNode_t 
# 1252
node, const T &
# 1253
symbol, const void *
# 1254
src, size_t 
# 1255
count, size_t 
# 1256
offset, cudaMemcpyKind 
# 1257
kind) 
# 1258
{ 
# 1259
return ::cudaGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec, node, (const void *)(&symbol), src, count, offset, kind); 
# 1260
} 
# 1308 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1309
cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t 
# 1310
hGraphExec, cudaGraphNode_t 
# 1311
node, void *
# 1312
dst, const T &
# 1313
symbol, size_t 
# 1314
count, size_t 
# 1315
offset, cudaMemcpyKind 
# 1316
kind) 
# 1317
{ 
# 1318
return ::cudaGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec, node, dst, (const void *)(&symbol), count, offset, kind); 
# 1319
} 
# 1347 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1348
cudaUserObjectCreate(cudaUserObject_t *
# 1349
object_out, T *
# 1350
objectToWrap, unsigned 
# 1351
initialRefcount, unsigned 
# 1352
flags) 
# 1353
{ 
# 1354
return ::cudaUserObjectCreate(object_out, objectToWrap, [](void *
# 1357
vpObj) { delete (reinterpret_cast< T *>(vpObj)); } , initialRefcount, flags); 
# 1360
} 
# 1362
template< class T> static inline cudaError_t 
# 1363
cudaUserObjectCreate(cudaUserObject_t *
# 1364
object_out, T *
# 1365
objectToWrap, unsigned 
# 1366
initialRefcount, cudaUserObjectFlags 
# 1367
flags) 
# 1368
{ 
# 1369
return cudaUserObjectCreate(object_out, objectToWrap, initialRefcount, (unsigned)flags); 
# 1370
} 
# 1397 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1398
cudaGetSymbolAddress(void **
# 1399
devPtr, const T &
# 1400
symbol) 
# 1402
{ 
# 1403
return ::cudaGetSymbolAddress(devPtr, (const void *)(&symbol)); 
# 1404
} 
# 1429 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1430
cudaGetSymbolSize(size_t *
# 1431
size, const T &
# 1432
symbol) 
# 1434
{ 
# 1435
return ::cudaGetSymbolSize(size, (const void *)(&symbol)); 
# 1436
} 
# 1473 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T, int dim, cudaTextureReadMode readMode> 
# 1474
__attribute((deprecated)) static inline cudaError_t cudaBindTexture(size_t *
# 1475
offset, const texture< T, dim, readMode>  &
# 1476
tex, const void *
# 1477
devPtr, const cudaChannelFormatDesc &
# 1478
desc, size_t 
# 1479
size = ((2147483647) * 2U) + 1U) 
# 1481
{ 
# 1482
return ::cudaBindTexture(offset, &tex, devPtr, &desc, size); 
# 1483
} 
# 1519 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T, int dim, cudaTextureReadMode readMode> 
# 1520
__attribute((deprecated)) static inline cudaError_t cudaBindTexture(size_t *
# 1521
offset, const texture< T, dim, readMode>  &
# 1522
tex, const void *
# 1523
devPtr, size_t 
# 1524
size = ((2147483647) * 2U) + 1U) 
# 1526
{ 
# 1527
return cudaBindTexture(offset, tex, devPtr, (tex.channelDesc), size); 
# 1528
} 
# 1576 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T, int dim, cudaTextureReadMode readMode> 
# 1577
__attribute((deprecated)) static inline cudaError_t cudaBindTexture2D(size_t *
# 1578
offset, const texture< T, dim, readMode>  &
# 1579
tex, const void *
# 1580
devPtr, const cudaChannelFormatDesc &
# 1581
desc, size_t 
# 1582
width, size_t 
# 1583
height, size_t 
# 1584
pitch) 
# 1586
{ 
# 1587
return ::cudaBindTexture2D(offset, &tex, devPtr, &desc, width, height, pitch); 
# 1588
} 
# 1635 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T, int dim, cudaTextureReadMode readMode> 
# 1636
__attribute((deprecated)) static inline cudaError_t cudaBindTexture2D(size_t *
# 1637
offset, const texture< T, dim, readMode>  &
# 1638
tex, const void *
# 1639
devPtr, size_t 
# 1640
width, size_t 
# 1641
height, size_t 
# 1642
pitch) 
# 1644
{ 
# 1645
return ::cudaBindTexture2D(offset, &tex, devPtr, &(tex.channelDesc), width, height, pitch); 
# 1646
} 
# 1678 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T, int dim, cudaTextureReadMode readMode> 
# 1679
__attribute((deprecated)) static inline cudaError_t cudaBindTextureToArray(const texture< T, dim, readMode>  &
# 1680
tex, cudaArray_const_t 
# 1681
array, const cudaChannelFormatDesc &
# 1682
desc) 
# 1684
{ 
# 1685
return ::cudaBindTextureToArray(&tex, array, &desc); 
# 1686
} 
# 1717 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T, int dim, cudaTextureReadMode readMode> 
# 1718
__attribute((deprecated)) static inline cudaError_t cudaBindTextureToArray(const texture< T, dim, readMode>  &
# 1719
tex, cudaArray_const_t 
# 1720
array) 
# 1722
{ 
# 1723
cudaChannelFormatDesc desc; 
# 1724
cudaError_t err = ::cudaGetChannelDesc(&desc, array); 
# 1726
return (err == (cudaSuccess)) ? cudaBindTextureToArray(tex, array, desc) : err; 
# 1727
} 
# 1759 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T, int dim, cudaTextureReadMode readMode> 
# 1760
__attribute((deprecated)) static inline cudaError_t cudaBindTextureToMipmappedArray(const texture< T, dim, readMode>  &
# 1761
tex, cudaMipmappedArray_const_t 
# 1762
mipmappedArray, const cudaChannelFormatDesc &
# 1763
desc) 
# 1765
{ 
# 1766
return ::cudaBindTextureToMipmappedArray(&tex, mipmappedArray, &desc); 
# 1767
} 
# 1798 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T, int dim, cudaTextureReadMode readMode> 
# 1799
__attribute((deprecated)) static inline cudaError_t cudaBindTextureToMipmappedArray(const texture< T, dim, readMode>  &
# 1800
tex, cudaMipmappedArray_const_t 
# 1801
mipmappedArray) 
# 1803
{ 
# 1804
cudaChannelFormatDesc desc; 
# 1805
cudaArray_t levelArray; 
# 1806
cudaError_t err = ::cudaGetMipmappedArrayLevel(&levelArray, mipmappedArray, 0); 
# 1808
if (err != (cudaSuccess)) { 
# 1809
return err; 
# 1810
}  
# 1811
err = ::cudaGetChannelDesc(&desc, levelArray); 
# 1813
return (err == (cudaSuccess)) ? cudaBindTextureToMipmappedArray(tex, mipmappedArray, desc) : err; 
# 1814
} 
# 1841 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T, int dim, cudaTextureReadMode readMode> 
# 1842
__attribute((deprecated)) static inline cudaError_t cudaUnbindTexture(const texture< T, dim, readMode>  &
# 1843
tex) 
# 1845
{ 
# 1846
return ::cudaUnbindTexture(&tex); 
# 1847
} 
# 1877 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T, int dim, cudaTextureReadMode readMode> 
# 1878
__attribute((deprecated)) static inline cudaError_t cudaGetTextureAlignmentOffset(size_t *
# 1879
offset, const texture< T, dim, readMode>  &
# 1880
tex) 
# 1882
{ 
# 1883
return ::cudaGetTextureAlignmentOffset(offset, &tex); 
# 1884
} 
# 1929 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 1930
cudaFuncSetCacheConfig(T *
# 1931
func, cudaFuncCache 
# 1932
cacheConfig) 
# 1934
{ 
# 1935
return ::cudaFuncSetCacheConfig((const void *)func, cacheConfig); 
# 1936
} 
# 1938
template< class T> static inline cudaError_t 
# 1939
cudaFuncSetSharedMemConfig(T *
# 1940
func, cudaSharedMemConfig 
# 1941
config) 
# 1943
{ 
# 1944
return ::cudaFuncSetSharedMemConfig((const void *)func, config); 
# 1945
} 
# 1977 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> inline cudaError_t 
# 1978
cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *
# 1979
numBlocks, T 
# 1980
func, int 
# 1981
blockSize, size_t 
# 1982
dynamicSMemSize) 
# 1983
{ 
# 1984
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, 0); 
# 1985
} 
# 2029 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> inline cudaError_t 
# 2030
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *
# 2031
numBlocks, T 
# 2032
func, int 
# 2033
blockSize, size_t 
# 2034
dynamicSMemSize, unsigned 
# 2035
flags) 
# 2036
{ 
# 2037
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, flags); 
# 2038
} 
# 2043
class __cudaOccupancyB2DHelper { 
# 2044
size_t n; 
# 2046
public: __cudaOccupancyB2DHelper(size_t n_) : n(n_) { } 
# 2047
size_t operator()(int) 
# 2048
{ 
# 2049
return n; 
# 2050
} 
# 2051
}; 
# 2099 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class UnaryFunction, class T> static inline cudaError_t 
# 2100
cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int *
# 2101
minGridSize, int *
# 2102
blockSize, T 
# 2103
func, UnaryFunction 
# 2104
blockSizeToDynamicSMemSize, int 
# 2105
blockSizeLimit = 0, unsigned 
# 2106
flags = 0) 
# 2107
{ 
# 2108
cudaError_t status; 
# 2111
int device; 
# 2112
cudaFuncAttributes attr; 
# 2115
int maxThreadsPerMultiProcessor; 
# 2116
int warpSize; 
# 2117
int devMaxThreadsPerBlock; 
# 2118
int multiProcessorCount; 
# 2119
int funcMaxThreadsPerBlock; 
# 2120
int occupancyLimit; 
# 2121
int granularity; 
# 2124
int maxBlockSize = 0; 
# 2125
int numBlocks = 0; 
# 2126
int maxOccupancy = 0; 
# 2129
int blockSizeToTryAligned; 
# 2130
int blockSizeToTry; 
# 2131
int blockSizeLimitAligned; 
# 2132
int occupancyInBlocks; 
# 2133
int occupancyInThreads; 
# 2134
size_t dynamicSMemSize; 
# 2140
if (((!minGridSize) || (!blockSize)) || (!func)) { 
# 2141
return cudaErrorInvalidValue; 
# 2142
}  
# 2148
status = ::cudaGetDevice(&device); 
# 2149
if (status != (cudaSuccess)) { 
# 2150
return status; 
# 2151
}  
# 2153
status = cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, device); 
# 2157
if (status != (cudaSuccess)) { 
# 2158
return status; 
# 2159
}  
# 2161
status = cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device); 
# 2165
if (status != (cudaSuccess)) { 
# 2166
return status; 
# 2167
}  
# 2169
status = cudaDeviceGetAttribute(&devMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device); 
# 2173
if (status != (cudaSuccess)) { 
# 2174
return status; 
# 2175
}  
# 2177
status = cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device); 
# 2181
if (status != (cudaSuccess)) { 
# 2182
return status; 
# 2183
}  
# 2185
status = cudaFuncGetAttributes(&attr, func); 
# 2186
if (status != (cudaSuccess)) { 
# 2187
return status; 
# 2188
}  
# 2190
funcMaxThreadsPerBlock = (attr.maxThreadsPerBlock); 
# 2196
occupancyLimit = maxThreadsPerMultiProcessor; 
# 2197
granularity = warpSize; 
# 2199
if (blockSizeLimit == 0) { 
# 2200
blockSizeLimit = devMaxThreadsPerBlock; 
# 2201
}  
# 2203
if (devMaxThreadsPerBlock < blockSizeLimit) { 
# 2204
blockSizeLimit = devMaxThreadsPerBlock; 
# 2205
}  
# 2207
if (funcMaxThreadsPerBlock < blockSizeLimit) { 
# 2208
blockSizeLimit = funcMaxThreadsPerBlock; 
# 2209
}  
# 2211
blockSizeLimitAligned = (((blockSizeLimit + (granularity - 1)) / granularity) * granularity); 
# 2213
for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) { 
# 2217
if (blockSizeLimit < blockSizeToTryAligned) { 
# 2218
blockSizeToTry = blockSizeLimit; 
# 2219
} else { 
# 2220
blockSizeToTry = blockSizeToTryAligned; 
# 2221
}  
# 2223
dynamicSMemSize = blockSizeToDynamicSMemSize(blockSizeToTry); 
# 2225
status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&occupancyInBlocks, func, blockSizeToTry, dynamicSMemSize, flags); 
# 2232
if (status != (cudaSuccess)) { 
# 2233
return status; 
# 2234
}  
# 2236
occupancyInThreads = (blockSizeToTry * occupancyInBlocks); 
# 2238
if (occupancyInThreads > maxOccupancy) { 
# 2239
maxBlockSize = blockSizeToTry; 
# 2240
numBlocks = occupancyInBlocks; 
# 2241
maxOccupancy = occupancyInThreads; 
# 2242
}  
# 2246
if (occupancyLimit == maxOccupancy) { 
# 2247
break; 
# 2248
}  
# 2249
}  
# 2257
(*minGridSize) = (numBlocks * multiProcessorCount); 
# 2258
(*blockSize) = maxBlockSize; 
# 2260
return status; 
# 2261
} 
# 2295 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class UnaryFunction, class T> static inline cudaError_t 
# 2296
cudaOccupancyMaxPotentialBlockSizeVariableSMem(int *
# 2297
minGridSize, int *
# 2298
blockSize, T 
# 2299
func, UnaryFunction 
# 2300
blockSizeToDynamicSMemSize, int 
# 2301
blockSizeLimit = 0) 
# 2302
{ 
# 2303
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, blockSizeLimit, 0); 
# 2304
} 
# 2341 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 2342
cudaOccupancyMaxPotentialBlockSize(int *
# 2343
minGridSize, int *
# 2344
blockSize, T 
# 2345
func, size_t 
# 2346
dynamicSMemSize = 0, int 
# 2347
blockSizeLimit = 0) 
# 2348
{ 
# 2349
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, 0); 
# 2350
} 
# 2379 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 2380
cudaOccupancyAvailableDynamicSMemPerBlock(size_t *
# 2381
dynamicSmemSize, T 
# 2382
func, int 
# 2383
numBlocks, int 
# 2384
blockSize) 
# 2385
{ 
# 2386
return ::cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, (const void *)func, numBlocks, blockSize); 
# 2387
} 
# 2438 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 2439
cudaOccupancyMaxPotentialBlockSizeWithFlags(int *
# 2440
minGridSize, int *
# 2441
blockSize, T 
# 2442
func, size_t 
# 2443
dynamicSMemSize = 0, int 
# 2444
blockSizeLimit = 0, unsigned 
# 2445
flags = 0) 
# 2446
{ 
# 2447
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, flags); 
# 2448
} 
# 2482 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 2483
cudaOccupancyMaxPotentialClusterSize(int *
# 2484
clusterSize, T *
# 2485
func, const cudaLaunchConfig_t *
# 2486
config) 
# 2487
{ 
# 2488
return ::cudaOccupancyMaxPotentialClusterSize(clusterSize, (const void *)func, config); 
# 2489
} 
# 2525 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 2526
cudaOccupancyMaxActiveClusters(int *
# 2527
numClusters, T *
# 2528
func, const cudaLaunchConfig_t *
# 2529
config) 
# 2530
{ 
# 2531
return ::cudaOccupancyMaxActiveClusters(numClusters, (const void *)func, config); 
# 2532
} 
# 2565 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> inline cudaError_t 
# 2566
cudaFuncGetAttributes(cudaFuncAttributes *
# 2567
attr, T *
# 2568
entry) 
# 2570
{ 
# 2571
return ::cudaFuncGetAttributes(attr, (const void *)entry); 
# 2572
} 
# 2627 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T> static inline cudaError_t 
# 2628
cudaFuncSetAttribute(T *
# 2629
entry, cudaFuncAttribute 
# 2630
attr, int 
# 2631
value) 
# 2633
{ 
# 2634
return ::cudaFuncSetAttribute((const void *)entry, attr, value); 
# 2635
} 
# 2659 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T, int dim> 
# 2660
__attribute((deprecated)) static inline cudaError_t cudaBindSurfaceToArray(const surface< T, dim>  &
# 2661
surf, cudaArray_const_t 
# 2662
array, const cudaChannelFormatDesc &
# 2663
desc) 
# 2665
{ 
# 2666
return ::cudaBindSurfaceToArray(&surf, array, &desc); 
# 2667
} 
# 2690 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
template< class T, int dim> 
# 2691
__attribute((deprecated)) static inline cudaError_t cudaBindSurfaceToArray(const surface< T, dim>  &
# 2692
surf, cudaArray_const_t 
# 2693
array) 
# 2695
{ 
# 2696
cudaChannelFormatDesc desc; 
# 2697
cudaError_t err = ::cudaGetChannelDesc(&desc, array); 
# 2699
return (err == (cudaSuccess)) ? cudaBindSurfaceToArray(surf, array, desc) : err; 
# 2700
} 
# 2711 "/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include/cuda_runtime.h" 3
#pragma GCC diagnostic pop
# 64 "CMakeCUDACompilerId.cu"
const char *info_compiler = ("INFO:compiler[NVIDIA]"); 
# 66
const char *info_simulate = ("INFO:simulate[GNU]"); 
# 329 "CMakeCUDACompilerId.cu"
const char info_version[] = {'I', 'N', 'F', 'O', ':', 'c', 'o', 'm', 'p', 'i', 'l', 'e', 'r', '_', 'v', 'e', 'r', 's', 'i', 'o', 'n', '[', (('0') + ((11 / 10000000) % 10)), (('0') + ((11 / 1000000) % 10)), (('0') + ((11 / 100000) % 10)), (('0') + ((11 / 10000) % 10)), (('0') + ((11 / 1000) % 10)), (('0') + ((11 / 100) % 10)), (('0') + ((11 / 10) % 10)), (('0') + (11 % 10)), '.', (('0') + ((8 / 10000000) % 10)), (('0') + ((8 / 1000000) % 10)), (('0') + ((8 / 100000) % 10)), (('0') + ((8 / 10000) % 10)), (('0') + ((8 / 1000) % 10)), (('0') + ((8 / 100) % 10)), (('0') + ((8 / 10) % 10)), (('0') + (8 % 10)), '.', (('0') + ((89 / 10000000) % 10)), (('0') + ((89 / 1000000) % 10)), (('0') + ((89 / 100000) % 10)), (('0') + ((89 / 10000) % 10)), (('0') + ((89 / 1000) % 10)), (('0') + ((89 / 100) % 10)), (('0') + ((89 / 10) % 10)), (('0') + (89 % 10)), ']', '\000'}; 
# 356 "CMakeCUDACompilerId.cu"
const char info_simulate_version[] = {'I', 'N', 'F', 'O', ':', 's', 'i', 'm', 'u', 'l', 'a', 't', 'e', '_', 'v', 'e', 'r', 's', 'i', 'o', 'n', '[', (('0') + ((8 / 10000000) % 10)), (('0') + ((8 / 1000000) % 10)), (('0') + ((8 / 100000) % 10)), (('0') + ((8 / 10000) % 10)), (('0') + ((8 / 1000) % 10)), (('0') + ((8 / 100) % 10)), (('0') + ((8 / 10) % 10)), (('0') + (8 % 10)), '.', (('0') + ((5 / 10000000) % 10)), (('0') + ((5 / 1000000) % 10)), (('0') + ((5 / 100000) % 10)), (('0') + ((5 / 10000) % 10)), (('0') + ((5 / 1000) % 10)), (('0') + ((5 / 100) % 10)), (('0') + ((5 / 10) % 10)), (('0') + (5 % 10)), ']', '\000'}; 
# 376
const char *info_platform = ("INFO:platform[Linux]"); 
# 377
const char *info_arch = ("INFO:arch[]"); 
# 381
const char *info_language_dialect_default = ("INFO:dialect_default[14]"); 
# 399
int main(int argc, char *argv[]) 
# 400
{ 
# 401
int require = 0; 
# 402
require += (info_compiler[argc]); 
# 403
require += (info_platform[argc]); 
# 405
require += (info_version[argc]); 
# 408
require += (info_simulate[argc]); 
# 411
require += (info_simulate_version[argc]); 
# 413
require += (info_language_dialect_default[argc]); 
# 414
(void)argv; 
# 415
return require; 
# 416
} 

# 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#define _NV_ANON_NAMESPACE _GLOBAL__N__eb8e314c_22_CMakeCUDACompilerId_cu_bd57c623
#ifdef _NV_ANON_NAMESPACE
#endif
# 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#include "CMakeCUDACompilerId.cudafe1.stub.c"
# 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#undef _NV_ANON_NAMESPACE
