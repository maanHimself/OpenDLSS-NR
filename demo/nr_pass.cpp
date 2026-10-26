#include "nr_pass.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "numeric.h"
#include "reference.h"

struct NrPass::ParamsBlock {
  uint32_t fullWidth, fullHeight, validWidth, validHeight;
  uint32_t sourceWidth, sourceHeight;
  float autoMask, paperWhite, colorStrength, intensity, localTone, localStructure, skinStructure, style;
  uint32_t historyValid, seed, nrEnabled, pad0;
  float blendScale, pad1;
  float styleExposure, styleContrast, styleGamma, styleSaturation;
  float styleHue, styleVibrance, styleStrength;
  uint32_t styleMode;
};

namespace {
struct UnpackPush { float background[16]; uint32_t width, height; };
}  // namespace

uint32_t NrPass::shaderReadOnlyLayout() { return (uint32_t)VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; }

NrPass::NrPass(const GpuDevice& device, uint32_t width, uint32_t height, const std::string& modelDir, const std::string& kernelDir,
               const std::string& demoShaderDir)
    : device_((VkDevice)device.device), width_(width), height_(height), demoShaderDir_(demoShaderDir) {
  context_ = std::make_unique<vk::Context>((VkInstance)device.instance, (VkPhysicalDevice)device.physicalDevice, device_,
                                           device.queueFamily, device.nrQueueIndex);
  fprintf(stderr, "[nr] context adopted (%s, queue %u.%u)\n", context_->deviceName().c_str(), device.queueFamily, device.nrQueueIndex);
  model_ = std::make_unique<nr::Model>(*context_, modelDir, false);
  fprintf(stderr, "[nr] model loaded\n");
  kernels_ = std::make_unique<nr::Kernels>(*context_, kernelDir);
  fprintf(stderr, "[nr] kernels loaded\n");
  kernels_->setSiluTable(ref::siluTable());
  const nr::Tensor& blend = model_->tensor(70, 0, "blend_scale");   // one f16 (production-pipeline.js unpack2x16float)
  blendScale_ = blend.byteLength >= 2 ? num::f16ToF32((uint16_t)(blend.bytes[0] | (blend.bytes[1] << 8))) : 1.0f;
  fprintf(stderr, "[nr] blend scale %g\n", blendScale_);

  VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerInfo.magFilter = samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.maxLod = 1.0f;
  VK_CHECK(vkCreateSampler(device_, &samplerInfo, nullptr, &linearSampler_));
  samplerInfo.magFilter = samplerInfo.minFilter = VK_FILTER_NEAREST;
  VK_CHECK(vkCreateSampler(device_, &samplerInfo, nullptr, &nearestSampler_));
  // the parameters are written into the command stream (vkCmdUpdateBuffer): device-local, one per parity
  for (vk::Buffer& b : params_)
    b = context_->createBuffer(sizeof(ParamsBlock), false, "nr params", VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  // ---- compute pipelines: preprocess / composite share one layout (7 bindings); the motion unpack has its own
  VkDescriptorSetLayoutBinding bindings[7] = {
      {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
  VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layoutInfo.bindingCount = 7; layoutInfo.pBindings = bindings;
  VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &computeSetLayout_));
  VkPipelineLayoutCreateInfo plInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plInfo.setLayoutCount = 1; plInfo.pSetLayouts = &computeSetLayout_;
  VK_CHECK(vkCreatePipelineLayout(device_, &plInfo, nullptr, &computeLayout_));
  preprocessPipeline_ = computePipeline(demoShaderDir + "/nr_preprocess.comp.spv", computeLayout_);
  compositePipeline_ = computePipeline(demoShaderDir + "/nr_composite.comp.spv", computeLayout_);
  VkDescriptorSetLayoutBinding unpackBindings[2] = {
      {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
  layoutInfo.bindingCount = 2; layoutInfo.pBindings = unpackBindings;
  VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &unpackSetLayout_));
  VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(UnpackPush)};
  plInfo.pSetLayouts = &unpackSetLayout_; plInfo.pushConstantRangeCount = 1; plInfo.pPushConstantRanges = &push;
  VK_CHECK(vkCreatePipelineLayout(device_, &plInfo, nullptr, &unpackLayout_));
  unpackPipeline_ = computePipeline(demoShaderDir + "/velocity_unpack.comp.spv", unpackLayout_);

  VkDescriptorPoolSize poolSizes[4] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
                                       {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 10}};
  VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  poolInfo.maxSets = 8; poolInfo.poolSizeCount = 4; poolInfo.pPoolSizes = poolSizes;
  VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &pool_));
  VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  alloc.descriptorPool = pool_; alloc.descriptorSetCount = 1; alloc.pSetLayouts = &computeSetLayout_;
  for (int h = 0; h < 2; ++h)
    for (int k = 0; k < 2; ++k) VK_CHECK(vkAllocateDescriptorSets(device_, &alloc, &computeSets_[h][k]));
  alloc.pSetLayouts = &unpackSetLayout_;
  VK_CHECK(vkAllocateDescriptorSets(device_, &alloc, &unpackSet_));
  for (VkQueryPool& q : queries_) q = context_->createTimestampPool(kStampCount);
  VkCommandPoolCreateInfo cmdPoolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cmdPoolInfo.queueFamilyIndex = context_->queueFamily();
  VK_CHECK(vkCreateCommandPool(device_, &cmdPoolInfo, nullptr, &commandPool_));
  VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocate.commandPool = commandPool_; allocate.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY; allocate.commandBufferCount = 4;
  VK_CHECK(vkAllocateCommandBuffers(device_, &allocate, &computeCommands_[0][0]));
  fprintf(stderr, "[nr] pipelines ready\n");
  createSized(width, height);
}

void NrPass::createSized(uint32_t width, uint32_t height) {
  width_ = width; height_ = height;
  geometry_ = nr::Geometry::fromValid(width, height);
  graph_ = std::make_unique<nr::Graph>(*context_, *model_, *kernels_, geometry_, nr::Graph::Options{});
  features_ = graph_->allocate("input features", geometry_.fullWidth * geometry_.fullHeight, 16, nr::Format::F32);
  {
    // a warm-up record + run: the graph allocates its activations (the head among them) at the first record, and
    // the driver compiles / loads every pipeline and CUDA module before the first real frame
    context_->fillZero(features_->buffer);
    VkCommandBuffer commands = context_->beginCommands();
    graph_->record(commands, *features_);
    context_->endAndSubmit(commands, true);
  }
  // ---- images (the scene color and the velocity are the renderer's; views onto them are made at the first record)
  // (TRANSFER_DST: each is cleared once below)
  sceneMotion_ = createImage(width, height, VK_FORMAT_R16G16B16A16_SFLOAT,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                             VK_IMAGE_ASPECT_COLOR_BIT);
  for (Image& h : history_)
    h = createImage(width, height, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT);
  output_ = createImage(width, height, VK_FORMAT_R8G8B8A8_UNORM,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT);
  {
    // the storage images live in GENERAL for their whole life
    VkCommandBuffer commands = context_->beginCommands();
    for (Image* image : {&sceneMotion_, &history_[0], &history_[1], &output_})
      transition(commands, *image, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                 VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    VkClearColorValue zero{}; VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    for (Image* image : {&sceneMotion_, &history_[0], &history_[1], &output_})
      vkCmdClearColorImage(commands, image->image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    context_->endAndSubmit(commands, true);
  }
  color_ = ExternalView{}; velocity_ = ExternalView{};
  for (auto& w : queriesWritten_) w = false;
  framesSinceReset_ = 0;
}

void NrPass::destroySized() {
  VK_CHECK(vkDeviceWaitIdle(device_));
  VK_CHECK(vkResetCommandPool(device_, commandPool_, 0));
  for (Image* image : {&sceneMotion_, &history_[0], &history_[1], &output_}) destroyImage(*image);
  for (ExternalView* external : {&color_, &velocity_}) {
    if (external->view) vkDestroyImageView(device_, external->view, nullptr);
    *external = ExternalView{};
  }
  features_ = nullptr;
  graph_.reset();   // frees the activations
  kernels_->releaseScratch();   // the next graph sizes the split-K scratch for its own shapes
}

void NrPass::fallBackToBarriers() {
  const nr::Kernels::ChainTimeouts timeouts = kernels_->chainTimeouts();
  fprintf(stderr, "[nr] %u chained wait(s) timed out, the first on %s: that frame was wrong; rebuilding with barriers\n",
          timeouts.waits, timeouts.counter.c_str());
  nr::Kernels::setChainEnabled(false);
  destroySized();
  createSized(width_, height_);
  kernels_->resetChainTimeouts();
}

void NrPass::resize(uint32_t width, uint32_t height) {
  if (width == width_ && height == height_) return;
  auto t0 = std::chrono::high_resolution_clock::now();
  destroySized();
  createSized(width, height);
  fprintf(stderr, "[nr] resized to %ux%u (full %ux%u) in %.0f ms\n", width, height, geometry_.fullWidth, geometry_.fullHeight,
          std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count());
}

NrPass::~NrPass() {
  vkDeviceWaitIdle(device_);
  destroySized();
  // (shader modules from Context::loadShaderModule are owned and destroyed by the Context)
  for (VkQueryPool q : queries_) vkDestroyQueryPool(device_, q, nullptr);
  vkDestroyCommandPool(device_, commandPool_, nullptr);
  vkDestroyPipeline(device_, preprocessPipeline_, nullptr);
  vkDestroyPipeline(device_, compositePipeline_, nullptr);
  vkDestroyPipeline(device_, unpackPipeline_, nullptr);
  vkDestroyPipelineLayout(device_, computeLayout_, nullptr);
  vkDestroyPipelineLayout(device_, unpackLayout_, nullptr);
  vkDestroyDescriptorSetLayout(device_, computeSetLayout_, nullptr);
  vkDestroyDescriptorSetLayout(device_, unpackSetLayout_, nullptr);
  vkDestroyDescriptorPool(device_, pool_, nullptr);
  for (vk::Buffer& b : params_) context_->destroyBuffer(b);
  vkDestroySampler(device_, linearSampler_, nullptr);
  vkDestroySampler(device_, nearestSampler_, nullptr);
  kernels_.reset(); model_.reset(); context_.reset();
}

NrPass::Image NrPass::createImage(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect) {
  Image image; image.format = format;
  VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  info.imageType = VK_IMAGE_TYPE_2D; info.format = format; info.extent = {width, height, 1};
  info.mipLevels = 1; info.arrayLayers = 1; info.samples = VK_SAMPLE_COUNT_1_BIT; info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = usage; info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VK_CHECK(vkCreateImage(device_, &info, nullptr, &image.image));
  VkMemoryRequirements req; vkGetImageMemoryRequirements(device_, image.image, &req);
  VkPhysicalDeviceMemoryProperties props; vkGetPhysicalDeviceMemoryProperties(context_->physical(), &props);
  uint32_t type = UINT32_MAX;
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
    if ((req.memoryTypeBits & (1u << i)) && (props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { type = i; break; }
  if (type == UINT32_MAX) throw std::runtime_error("no device-local memory for an image");
  VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; alloc.allocationSize = req.size; alloc.memoryTypeIndex = type;
  VK_CHECK(vkAllocateMemory(device_, &alloc, nullptr, &image.memory));
  VK_CHECK(vkBindImageMemory(device_, image.image, image.memory, 0));
  VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  viewInfo.image = image.image; viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D; viewInfo.format = format;
  viewInfo.subresourceRange = {aspect, 0, 1, 0, 1};
  VK_CHECK(vkCreateImageView(device_, &viewInfo, nullptr, &image.view));
  return image;
}

void NrPass::destroyImage(Image& image) {
  if (image.view) vkDestroyImageView(device_, image.view, nullptr);
  if (image.image) vkDestroyImage(device_, image.image, nullptr);
  if (image.memory) vkFreeMemory(device_, image.memory, nullptr);
  image = Image{};
}

bool NrPass::bindExternal(ExternalView& external, const GpuImage& image) {
  VkImage handle = (VkImage)(uintptr_t)image.image;
  if (handle == external.image && (VkFormat)image.format == external.format) return false;
  if (external.view) vkDestroyImageView(device_, external.view, nullptr);
  external = ExternalView{};
  external.image = handle; external.format = (VkFormat)image.format;
  VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  viewInfo.image = handle; viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D; viewInfo.format = external.format;
  viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VK_CHECK(vkCreateImageView(device_, &viewInfo, nullptr, &external.view));
  return true;
}

void NrPass::transition(VkCommandBuffer commands, Image& image, VkImageLayout layout, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                        VkPipelineStageFlags dstStage, VkAccessFlags dstAccess, VkImageAspectFlags aspect) {
  VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.srcAccessMask = srcAccess; barrier.dstAccessMask = dstAccess;
  barrier.oldLayout = image.layout; barrier.newLayout = layout;
  barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image.image; barrier.subresourceRange = {aspect, 0, 1, 0, 1};
  vkCmdPipelineBarrier(commands, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
  image.layout = layout;
}

void NrPass::barrier(VkCommandBuffer commands, VkImage image, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags srcStage,
                     VkAccessFlags srcAccess, VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) {
  VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.srcAccessMask = srcAccess; barrier.dstAccessMask = dstAccess;
  barrier.oldLayout = from; barrier.newLayout = to;
  barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image; barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(commands, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

VkPipeline NrPass::computePipeline(const std::string& spv, VkPipelineLayout layout) {
  VkShaderModule module = context_->loadShaderModule(spv);
  VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
  info.layout = layout;
  VkPipeline pipeline;
  VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline));
  return pipeline;
}

void NrPass::updateComputeSets() {
  for (int h = 0; h < 2; ++h) {
    for (int k = 0; k < 2; ++k) {
      VkDescriptorImageInfo color{nearestSampler_, color_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      VkDescriptorImageInfo prev{linearSampler_, history_[h].view, VK_IMAGE_LAYOUT_GENERAL};
      VkDescriptorImageInfo motion{nearestSampler_, sceneMotion_.view, VK_IMAGE_LAYOUT_GENERAL};
      const nr::Activation& head = graph_->head();
      VkDescriptorBufferInfo buffer{k == 0 ? features_->buffer.buffer : head.buffer.buffer, 0, VK_WHOLE_SIZE};
      VkDescriptorBufferInfo params{params_[h].buffer, 0, sizeof(ParamsBlock)};
      VkDescriptorImageInfo out{VK_NULL_HANDLE, output_.view, VK_IMAGE_LAYOUT_GENERAL};
      VkDescriptorImageInfo next{VK_NULL_HANDLE, history_[1 - h].view, VK_IMAGE_LAYOUT_GENERAL};
      VkWriteDescriptorSet writes[7]{};
      for (int i = 0; i < 7; ++i) { writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet = computeSets_[h][k]; writes[i].dstBinding = i; writes[i].descriptorCount = 1; }
      writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; writes[0].pImageInfo = &color;
      writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; writes[1].pImageInfo = &prev;
      writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; writes[2].pImageInfo = &motion;
      writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[3].pBufferInfo = &buffer;
      writes[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; writes[4].pBufferInfo = &params;
      writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[5].pImageInfo = &out;
      writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[6].pImageInfo = &next;
      vkUpdateDescriptorSets(device_, 7, writes, 0, nullptr);
    }
  }
  VkDescriptorImageInfo velocity{nearestSampler_, velocity_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkDescriptorImageInfo motionOut{VK_NULL_HANDLE, sceneMotion_.view, VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet writes[2]{};
  for (int i = 0; i < 2; ++i) { writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet = unpackSet_; writes[i].dstBinding = i; writes[i].descriptorCount = 1; }
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; writes[0].pImageInfo = &velocity;
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[1].pImageInfo = &motionOut;
  vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);
}

NrPass::Frame NrPass::beginFrame(const NrControls& controls, const float background[16]) {
  historyIndex_ = frames_ & 1u;
  // the timestamps of the frame that used this parity last (two frames ago), when the GPU is done with them
  if (queriesWritten_[historyIndex_].load()) {
    uint64_t ticks[kStampCount * 2];
    VkResult r = vkGetQueryPoolResults(device_, queries_[historyIndex_], 0, kStampCount, sizeof(ticks), ticks, 16,
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    bool available = r == VK_SUCCESS;
    for (int i = 0; i < kStampCount && available; ++i) available = ticks[i * 2 + 1] != 0;
    if (available) {
      const double period = context_->timestampPeriodNs() * 1e-6;
      auto ms = [&](Stamp s) { return ticks[s * 2] * period; };
      timings_.sceneMs = ms(kSceneEnd) - ms(kFrameStart);
      timings_.preprocessMs = ms(kPreprocessEnd) - ms(kSceneEnd);
      timings_.networkMs = ms(kNetworkEnd) - ms(kPreprocessEnd);
      timings_.compositeMs = ms(kCompositeEnd) - ms(kNetworkEnd);
      timings_.presentMs = ms(kPresentEnd) - ms(kCompositeEnd);
      timings_.frameMs = ms(kPresentEnd) - ms(kFrameStart);
    }
  }
  Frame frame;
  frame.parity = historyIndex_;
  frame.enabled = controls.enabled;
  memcpy(frame.background, background, sizeof(frame.background));
  ParamsBlock block{};
  block.fullWidth = geometry_.fullWidth; block.fullHeight = geometry_.fullHeight;
  block.validWidth = width_; block.validHeight = height_;
  block.sourceWidth = width_; block.sourceHeight = height_;
  block.autoMask = controls.autoMask ? 1.0f : 0.0f;
  block.paperWhite = controls.paperWhite; block.colorStrength = controls.colorStrength; block.intensity = controls.intensity;
  block.localTone = controls.localTone; block.localStructure = controls.localStructure; block.skinStructure = controls.skinStructure;
  block.style = (float)controls.style;
  block.historyValid = (controls.temporal && framesSinceReset_ > 0) ? 1u : 0u;
  block.seed = frames_;
  block.nrEnabled = controls.enabled ? 1u : 0u;
  block.blendScale = blendScale_;
  block.styleExposure = controls.styleExposure; block.styleContrast = controls.styleContrast; block.styleGamma = controls.styleGamma;
  block.styleSaturation = controls.styleSaturation; block.styleHue = controls.styleHue; block.styleVibrance = controls.styleVibrance;
  block.styleStrength = controls.styleStrength; block.styleMode = controls.styleCustom ? 1u : 0u;
  static_assert(sizeof(ParamsBlock) <= sizeof(frame.params));
  memcpy(frame.params, &block, sizeof(block));
  return frame;
}

void NrPass::writeStamp(VkCommandBuffer commands, uint32_t parity, Stamp stamp, VkPipelineStageFlagBits stage) {
  if (stamp == kFrameStart) vkCmdResetQueryPool(commands, queries_[parity], 0, kStampCount);
  vkCmdWriteTimestamp(commands, stage, queries_[parity], stamp);
  if (stamp == kPresentEnd) queriesWritten_[parity] = true;
}

void NrPass::recordStamp(void* commandBuffer, const Frame& frame, Stamp stamp) {
  writeStamp((VkCommandBuffer)commandBuffer, frame.parity, stamp);
}

// The NR command buffers are recorded once: the graph's ~250 dispatches with their descriptor sets cost ~3.7 ms of
// CPU per record, and nothing in them changes between frames (the parameters live in a per-parity UBO written in
// the command stream, the history alternates between two fixed images by parity).
void NrPass::buildComputeCommands() {
  VK_CHECK(vkResetCommandPool(device_, commandPool_, 0));
  for (uint32_t h = 0; h < 2; ++h) {
    // the graph's descriptor sets of this parity live in the Context's pool h for the lifetime of the command buffer
    context_->resetDescriptorPool(h);
    for (int enabled = 0; enabled < 2; ++enabled) {
      VkCommandBuffer commands = computeCommands_[h][enabled];
      VkCommandBufferInheritanceInfo inheritance{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
      VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
      begin.pInheritanceInfo = &inheritance;
      VK_CHECK(vkBeginCommandBuffer(commands, &begin));
      if (enabled) {
        // preprocess -> features
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, preprocessPipeline_);
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, computeLayout_, 0, 1, &computeSets_[h][0], 0, nullptr);
        vkCmdDispatch(commands, (geometry_.fullWidth + 7) / 8, (geometry_.fullHeight + 7) / 8, 1);
        context_->computeBarrier(commands);
      }
      writeStamp(commands, h, kPreprocessEnd);
      if (enabled) graph_->record(commands, *features_);   // the network
      writeStamp(commands, h, kNetworkEnd);
      // composite -> output (rgba8) + next history
      vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, compositePipeline_);
      vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, computeLayout_, 0, 1, &computeSets_[h][1], 0, nullptr);
      vkCmdDispatch(commands, (width_ + 7) / 8, (height_ + 7) / 8, 1);
      // output: compute write -> the copy to the renderer's texture; next history: compute write -> next frame's reads
      VkMemoryBarrier memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      memory.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; memory.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
      vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0, 1, &memory, 0, nullptr, 0, nullptr);
      writeStamp(commands, h, kCompositeEnd);
      VK_CHECK(vkEndCommandBuffer(commands));
    }
  }
}

void NrPass::record(void* commandBuffer, const Frame& frame, const GpuImage& color, const GpuImage& velocity, const GpuImage& output) {
  VkCommandBuffer commands = (VkCommandBuffer)commandBuffer;
  const uint32_t h = frame.parity;
  // views onto the renderer's images; when they changed (first frame, resize) the descriptor sets and the
  // pre-recorded command buffers that reference them are rebuilt (nothing of them is pending then)
  bool changed = bindExternal(color_, color);
  changed = bindExternal(velocity_, velocity) || changed;
  if (changed) {
    updateComputeSets();
    buildComputeCommands();
  }
  // this frame's parameters, in the command stream
  vkCmdUpdateBuffer(commands, params_[h].buffer, 0, sizeof(ParamsBlock), frame.params);
  VkMemoryBarrier paramsBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  paramsBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; paramsBarrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
  vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &paramsBarrier, 0, nullptr, 0, nullptr);
  // the scene color and the velocity: the renderer's attachment writes -> our compute reads
  barrier(commands, color_.image, (VkImageLayout)color.layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
          VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
  barrier(commands, velocity_.image, (VkImageLayout)velocity.layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
          VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
  writeStamp(commands, h, kSceneEnd);
  // motion vectors: (id, depth, motion bits) -> rgba16f: current -> previous in uv units, and whether it is on screen
  vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, unpackPipeline_);
  vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, unpackLayout_, 0, 1, &unpackSet_, 0, nullptr);
  UnpackPush push{};
  memcpy(push.background, frame.background, sizeof(push.background));
  push.width = width_; push.height = height_;
  vkCmdPushConstants(commands, unpackLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
  vkCmdDispatch(commands, (width_ + 7) / 8, (height_ + 7) / 8, 1);
  // the motion image is written as a storage image and read through a sampler (computeBarrier covers storage reads only)
  VkMemoryBarrier motionBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  motionBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; motionBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &motionBarrier, 0, nullptr, 0, nullptr);
  // preprocess -> network -> composite (pre-recorded)
  vkCmdExecuteCommands(commands, 1, &computeCommands_[h][frame.enabled ? 1 : 0]);
  // the composited output -> the renderer's texture (sampled by its present view)
  VkImage target = (VkImage)(uintptr_t)output.image;
  barrier(commands, target, (VkImageLayout)output.layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
          VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
  barrier(commands, output_.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
  VkImageCopy region{};
  region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.extent = {width_, height_, 1};
  vkCmdCopyImage(commands, output_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  barrier(commands, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
  barrier(commands, output_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
}

std::vector<uint8_t> NrPass::readImage(VkImage image, uint32_t bytesPerPixel, VkImageLayout layout) {
  VK_CHECK(vkDeviceWaitIdle(device_));
  vk::Buffer staging = context_->createBuffer((VkDeviceSize)width_ * height_ * bytesPerPixel, true, "capture");
  VkCommandBuffer commands = context_->beginCommands();
  barrier(commands, image, layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
  VkBufferImageCopy region{};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = {width_, height_, 1};
  vkCmdCopyImageToBuffer(commands, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging.buffer, 1, &region);
  barrier(commands, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layout, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
  context_->endAndSubmit(commands, true);
  std::vector<uint8_t> bytes((const uint8_t*)staging.mapped, (const uint8_t*)staging.mapped + (size_t)width_ * height_ * bytesPerPixel);
  context_->destroyBuffer(staging);
  return bytes;
}

void NrPass::saveOutput(const std::string& path, bool sceneInstead) {
  if (sceneInstead && !color_.image) return;
  std::vector<uint8_t> bytes = sceneInstead ? readImage(color_.image, 8, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                                            : readImage(output_.image, 4, VK_IMAGE_LAYOUT_GENERAL);
  std::ofstream file(path, std::ios::binary);
  file << "P6\n" << width_ << " " << height_ << "\n255\n";
  std::vector<uint8_t> row(width_ * 3);
  for (uint32_t y = 0; y < height_; ++y) {
    for (uint32_t x = 0; x < width_; ++x) {
      if (sceneInstead) {
        const uint16_t* half = (const uint16_t*)(bytes.data() + ((size_t)y * width_ + x) * 8);
        for (int c = 0; c < 3; ++c) {
          float v = num::f16ToF32(half[c]);
          v = v / (1.0f + v);  // a plain Reinhard view of the HDR scene
          v = v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
          row[x * 3 + c] = (uint8_t)std::lround(std::fmin(std::fmax(v, 0.0f), 1.0f) * 255.0f);
        }
      } else {
        const uint8_t* px = bytes.data() + ((size_t)y * width_ + x) * 4;
        row[x * 3] = px[0]; row[x * 3 + 1] = px[1]; row[x * 3 + 2] = px[2];
      }
    }
    file.write((const char*)row.data(), row.size());
  }
  if (!sceneInstead) {
    // the network's temporal blend weights (head channel 3 -> sigmoid x blend_scale) over the valid area
    const nr::Activation& head = graph_->head();
    std::vector<uint8_t> raw = context_->download(head.buffer, head.validBytes(), 0);
    const float* values = (const float*)raw.data();
    double sum = 0, low = 0, high = 0; size_t count = 0;
    for (uint32_t y = 0; y < height_; ++y)
      for (uint32_t x = 0; x < width_; ++x) {
        float w = values[((size_t)y * geometry_.fullWidth + x) * 4 + 3];
        float weight = std::fmin(std::fmax((1.0f / (1.0f + std::exp(-w))) * blendScale_, 0.0f), 1.0f);
        sum += weight; count++; if (weight < 0.1f) low++; if (weight > 0.5f) high++;
      }
    fprintf(stderr, "[nr] %s: mean history weight %.3f (blend scale %.3f), <0.1: %.1f%%, >0.5: %.1f%%\n", path.c_str(), sum / count, blendScale_,
            100.0 * low / count, 100.0 * high / count);
  }
}

void NrPass::saveRaw(const std::string& path, int kind) {
  if (kind == 0 && !color_.image) return;
  if (kind == 2) {
    // the renderer's velocity buffer as is: rgba32ui per pixel (object id, depth bits, motion x bits, motion y bits)
    if (!velocity_.image) return;
    std::vector<uint8_t> bytes = readImage(velocity_.image, 16, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    std::ofstream file(path, std::ios::binary);
    uint32_t header[3] = {width_, height_, 4};
    file.write((const char*)header, sizeof(header));
    file.write((const char*)bytes.data(), bytes.size());
    return;
  }
  const uint32_t bytesPerPixel = 8, channels = 3;   // rgba16f both: scene rgb; motion x, y and history 1 / 0
  std::vector<uint8_t> bytes = readImage(kind == 0 ? color_.image : sceneMotion_.image, bytesPerPixel,
                                         kind == 0 ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL);
  std::vector<float> out((size_t)width_ * height_ * channels);
  const uint16_t* half = (const uint16_t*)bytes.data();
  for (size_t i = 0; i < (size_t)width_ * height_; ++i)
    for (uint32_t c = 0; c < channels; ++c) out[i * channels + c] = num::f16ToF32(half[i * (bytesPerPixel / 2) + c]);
  std::ofstream file(path, std::ios::binary);
  uint32_t header[3] = {width_, height_, channels};
  file.write((const char*)header, sizeof(header));
  file.write((const char*)out.data(), out.size() * sizeof(float));
}
