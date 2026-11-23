// GPU verification of direct-binding extensions:
// 1. GLCompute OpImageWrite to texture2d_array<uint>, texture2d_array<float>, and texture2d<uint>.
// 2. Direct-binding Fragment shader with OpImageFetch + BufferBlock + PushConstant -> FragDepth + FragStencilRefEXT.
// 3. Direct-binding Fragment shader with StorageClassPrivate + Float16 half4 bit-exact color output.
#import <Metal/Metal.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <spirv2air.h>
#include <vector>

static std::vector<uint32_t> readWords(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  std::vector<uint32_t> words(file ? size_t(file.tellg()) / 4 : 0);
  if (words.size() < 5) return {};
  file.seekg(0);
  file.read(reinterpret_cast<char*>(words.data()), words.size() * 4);
  return words;
}

static id<MTLLibrary> compileSpirv(id<MTLDevice> device, const char* path, const spirv2air::Options& options) {
  auto words = readWords(path);
  if (words.empty()) {
    std::fprintf(stderr, "cannot read %s\n", path);
    return nil;
  }
  auto result = spirv2air::translate(words.data(), words.size(), options);
  if (!result.ok()) {
    std::fprintf(stderr, "spirv2air(%s): %s\n", path, result.error.c_str());
    return nil;
  }
  dispatch_data_t blob = dispatch_data_create(result.metallib.data(), result.metallib.size(), nullptr, nullptr);
  NSError* error = nil;
  id<MTLLibrary> lib = [device newLibraryWithData:blob error:&error];
  if (!lib) {
    std::fprintf(stderr, "newLibraryWithData(%s): %s\n", path, error.description.UTF8String);
    return nil;
  }
  return lib;
}

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc != 4) return 2;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return 77;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSError* error = nil;

    // ---- Test 1: GLCompute OpImageWrite
    {
      spirv2air::Options opts;
      opts.bindings = {
          {0, 0, 0, 0},
          {0, 1, 0, 1},
          {0, 2, 0, 2},
      };
      id<MTLLibrary> lib = compileSpirv(device, argv[1], opts);
      if (!lib) return 1;
      id<MTLComputePipelineState> pipe =
          [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"main"] error:&error];
      if (!pipe) return std::fprintf(stderr, "compute pipe: %s\n", error.description.UTF8String), 1;

      MTLTextureDescriptor* d_uarr = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG32Uint
                                                                                        width:4
                                                                                       height:4
                                                                                    mipmapped:NO];
      d_uarr.textureType = MTLTextureType2DArray;
      d_uarr.arrayLength = 2;
      d_uarr.usage = MTLTextureUsageShaderWrite;
      d_uarr.storageMode = MTLStorageModeShared;
      id<MTLTexture> t_uarr = [device newTextureWithDescriptor:d_uarr];

      MTLTextureDescriptor* d_farr = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                        width:4
                                                                                       height:4
                                                                                    mipmapped:NO];
      d_farr.textureType = MTLTextureType2DArray;
      d_farr.arrayLength = 2;
      d_farr.usage = MTLTextureUsageShaderWrite;
      d_farr.storageMode = MTLStorageModeShared;
      id<MTLTexture> t_farr = [device newTextureWithDescriptor:d_farr];

      MTLTextureDescriptor* d_u2d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
                                                                                       width:4
                                                                                      height:4
                                                                                   mipmapped:NO];
      d_u2d.usage = MTLTextureUsageShaderWrite;
      d_u2d.storageMode = MTLStorageModeShared;
      id<MTLTexture> t_u2d = [device newTextureWithDescriptor:d_u2d];

      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pipe];
      [enc setTexture:t_uarr atIndex:0];
      [enc setTexture:t_farr atIndex:1];
      [enc setTexture:t_u2d atIndex:2];
      [enc dispatchThreads:MTLSizeMake(4, 4, 1) threadsPerThreadgroup:MTLSizeMake(4, 4, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.error) return std::fprintf(stderr, "image_write cb: %s\n", cb.error.description.UTF8String), 1;

      uint32_t uarr[4 * 4 * 2] = {};
      float farr[4 * 4] = {};
      uint32_t u2d[4 * 4] = {};
      [t_uarr getBytes:uarr bytesPerRow:4 * 8 bytesPerImage:4 * 4 * 8 fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 slice:1];
      [t_farr getBytes:farr bytesPerRow:4 * 4 bytesPerImage:4 * 4 * 4 fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 slice:1];
      [t_u2d getBytes:u2d bytesPerRow:4 * 4 fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
      for (uint32_t y = 0; y < 4; ++y) {
        for (uint32_t x = 0; x < 4; ++x) {
          uint32_t idx = y * 4 + x;
          if (uarr[idx * 2] != x + 10 || uarr[idx * 2 + 1] != y + 100)
            return std::fprintf(stderr, "uarr(%u,%u) = (%u,%u)\n", x, y, uarr[idx * 2], uarr[idx * 2 + 1]), 1;
          float exp_f = float(x + y) * 0.25f;
          if (farr[idx] != exp_f) return std::fprintf(stderr, "farr(%u,%u) = %g != %g\n", x, y, farr[idx], exp_f), 1;
          if (u2d[idx] != x + 10 + y + 100)
            return std::fprintf(stderr, "u2d(%u,%u) = %u\n", x, y, u2d[idx]), 1;
        }
      }
    }

    // Shared fullscreen vertex shader.
    NSString* vs_src = @R"(
      #include <metal_stdlib>
      using namespace metal;
      vertex float4 fs_vs(uint vid [[vertex_id]]) {
        const float2 xy[3] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
        return float4(xy[vid], 0, 1);
      }
    )";
    id<MTLLibrary> vs_lib = [device newLibraryWithSource:vs_src options:nil error:&error];
    if (!vs_lib) return std::fprintf(stderr, "vs: %s\n", error.description.UTF8String), 1;
    id<MTLFunction> vs_fn = [vs_lib newFunctionWithName:@"fs_vs"];

    // ---- Test 2: Direct-binding Fragment with FragDepth + FragStencilRefEXT
    {
      spirv2air::Options opts;
      opts.directBindings = true;
      opts.pushConstantBuffer = 0;
      opts.bindings = {
          {0, 0, 1, 0},
          {1, 0, 0, 2},
      };
      id<MTLLibrary> lib = compileSpirv(device, argv[2], opts);
      if (!lib) return 1;

      MTLRenderPipelineDescriptor* pdesc = [MTLRenderPipelineDescriptor new];
      pdesc.vertexFunction = vs_fn;
      pdesc.fragmentFunction = [lib newFunctionWithName:@"main"];
      pdesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
      pdesc.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
      id<MTLRenderPipelineState> pipe = [device newRenderPipelineStateWithDescriptor:pdesc error:&error];
      if (!pipe) return std::fprintf(stderr, "stencil pipe: %s\n", error.description.UTF8String), 1;

      MTLDepthStencilDescriptor* ds_desc = [MTLDepthStencilDescriptor new];
      ds_desc.depthCompareFunction = MTLCompareFunctionAlways;
      ds_desc.depthWriteEnabled = YES;
      MTLStencilDescriptor* st = [MTLStencilDescriptor new];
      st.stencilCompareFunction = MTLCompareFunctionAlways;
      st.depthStencilPassOperation = MTLStencilOperationReplace;
      st.writeMask = 0xFF;
      ds_desc.frontFaceStencil = st;
      ds_desc.backFaceStencil = st;
      id<MTLDepthStencilState> ds_state = [device newDepthStencilStateWithDescriptor:ds_desc];

      MTLTextureDescriptor* src_desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
                                                                                          width:4
                                                                                         height:4
                                                                                      mipmapped:NO];
      src_desc.usage = MTLTextureUsageShaderRead;
      src_desc.storageMode = MTLStorageModeShared;
      id<MTLTexture> src_tex = [device newTextureWithDescriptor:src_desc];
      uint32_t src_data[16];
      for (uint32_t i = 0; i < 16; ++i) src_data[i] = i * 3 + 1;
      [src_tex replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:src_data bytesPerRow:16];

      id<MTLBuffer> buf = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
      uint32_t* bptr = static_cast<uint32_t*>(buf.contents);
      for (uint32_t i = 0; i < 16; ++i) bptr[i] = 20 + i;

      MTLTextureDescriptor* ds_tex_desc =
          [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                                             width:4
                                                            height:4
                                                         mipmapped:NO];
      ds_tex_desc.usage = MTLTextureUsageRenderTarget;
      ds_tex_desc.storageMode = MTLStorageModePrivate;
      id<MTLTexture> ds_tex = [device newTextureWithDescriptor:ds_tex_desc];

      id<MTLBuffer> stencil_out = [device newBufferWithLength:16 options:MTLResourceStorageModeShared];
      id<MTLBuffer> depth_out = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];

      uint32_t pc[4] = {7, 0, 0, 0};
      MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
      rp.depthAttachment.texture = ds_tex;
      rp.depthAttachment.loadAction = MTLLoadActionClear;
      rp.depthAttachment.storeAction = MTLStoreActionStore;
      rp.depthAttachment.clearDepth = 0.0;
      rp.stencilAttachment.texture = ds_tex;
      rp.stencilAttachment.loadAction = MTLLoadActionClear;
      rp.stencilAttachment.storeAction = MTLStoreActionStore;
      rp.stencilAttachment.clearStencil = 0;

      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
      [enc setRenderPipelineState:pipe];
      [enc setDepthStencilState:ds_state];
      [enc setFragmentBytes:pc length:sizeof(pc) atIndex:0];
      [enc setFragmentBuffer:buf offset:0 atIndex:1];
      [enc setFragmentTexture:src_tex atIndex:2];
      [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [enc endEncoding];

      id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
      [blit copyFromTexture:ds_tex sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                 sourceSize:MTLSizeMake(4, 4, 1) toBuffer:depth_out destinationOffset:0
           destinationBytesPerRow:16 destinationBytesPerImage:64 options:MTLBlitOptionDepthFromDepthStencil];
      [blit copyFromTexture:ds_tex sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                 sourceSize:MTLSizeMake(4, 4, 1) toBuffer:stencil_out destinationOffset:0
           destinationBytesPerRow:4 destinationBytesPerImage:16 options:MTLBlitOptionStencilFromDepthStencil];
      [blit endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.error) return std::fprintf(stderr, "stencil cb: %s\n", cb.error.description.UTF8String), 1;

      const float* depths = static_cast<const float*>(depth_out.contents);
      const uint8_t* stencils = static_cast<const uint8_t*>(stencil_out.contents);
      for (uint32_t y = 0; y < 4; ++y) {
        for (uint32_t x = 0; x < 4; ++x) {
          uint32_t idx = y * 4 + x;
          float exp_d = (float(x) + 0.5f) * 0.0625f;
          uint8_t exp_s = uint8_t((src_data[idx] + bptr[x + y] + pc[0]) & 0xFFu);
          if (std::fabs(depths[idx] - exp_d) > 1e-5f)
            return std::fprintf(stderr, "depth(%u,%u) = %g != %g\n", x, y, depths[idx], exp_d), 1;
          if (stencils[idx] != exp_s)
            return std::fprintf(stderr, "stencil(%u,%u) = %u != %u\n", x, y, stencils[idx], exp_s), 1;
        }
      }
    }

    // ---- Test 3: Direct-binding Fragment with Float16 half4 bit-exact color output
    {
      spirv2air::Options opts;
      opts.directBindings = true;
      opts.pushConstantBuffer = 0;
      id<MTLLibrary> lib = compileSpirv(device, argv[3], opts);
      if (!lib) return 1;

      MTLRenderPipelineDescriptor* pdesc = [MTLRenderPipelineDescriptor new];
      pdesc.vertexFunction = vs_fn;
      pdesc.fragmentFunction = [lib newFunctionWithName:@"main"];
      pdesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
      id<MTLRenderPipelineState> pipe = [device newRenderPipelineStateWithDescriptor:pdesc error:&error];
      if (!pipe) return std::fprintf(stderr, "half pipe: %s\n", error.description.UTF8String), 1;

      MTLTextureDescriptor* rt_desc =
          [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                             width:2
                                                            height:2
                                                         mipmapped:NO];
      rt_desc.usage = MTLTextureUsageRenderTarget;
      rt_desc.storageMode = MTLStorageModeShared;
      id<MTLTexture> rt = [device newTextureWithDescriptor:rt_desc];

      // 1.0h, -2.5h, 0x0001 (smallest positive subnormal), 0x8000 (-0.0h)
      uint32_t pc[4] = {0x3C00u, 0xC100u, 0x0001u, 0x8000u};
      MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
      rp.colorAttachments[0].texture = rt;
      rp.colorAttachments[0].loadAction = MTLLoadActionClear;
      rp.colorAttachments[0].storeAction = MTLStoreActionStore;

      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
      [enc setRenderPipelineState:pipe];
      [enc setFragmentBytes:pc length:sizeof(pc) atIndex:0];
      [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.error) return std::fprintf(stderr, "half cb: %s\n", cb.error.description.UTF8String), 1;

      uint16_t pixels[4 * 4] = {};
      [rt getBytes:pixels bytesPerRow:2 * 8 fromRegion:MTLRegionMake2D(0, 0, 2, 2) mipmapLevel:0];
      for (uint32_t p = 0; p < 4; ++p) {
        for (uint32_t c = 0; c < 4; ++c) {
          if (pixels[p * 4 + c] != uint16_t(pc[c]))
            return std::fprintf(stderr, "half pixel %u ch %u = 0x%04X != 0x%04X\n", p, c, pixels[p * 4 + c], pc[c]), 1;
        }
      }
    }

    std::printf("PASS: image_write + direct_stencil + direct_half\n");
    return 0;
  }
}
