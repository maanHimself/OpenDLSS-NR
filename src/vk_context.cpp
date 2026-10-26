#include "vk_context.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>

namespace vk {

namespace {
constexpr VkDeviceSize kStagingBytes = 256ull << 20;  // 256 MiB staging window
constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";
std::atomic<uint32_t> g_validationErrors{0};

bool envFlag(const char* name) {
  const char* value = getenv(name);
  return value && *value && strcmp(value, "0") != 0;
}

VkBool32 VKAPI_PTR onDebugMessage(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT type,
                                  const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
  if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) && (type & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT))
    ++g_validationErrors;
  fprintf(stderr, "[vk] %s\n", data->pMessage ? data->pMessage : "");
  return VK_FALSE;
}
}  // namespace

uint32_t Context::validationErrors() { return g_validationErrors.load(); }

Context::Context() {
  if (volkInitialize() != VK_SUCCESS) throw std::runtime_error("vulkan-1.dll unavailable");

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "dlss5-vulkan";
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instanceInfo.pApplicationInfo = &app;
  validation_ = envFlag("DLSS5VK_VALIDATION");
  const bool debug = envFlag("DLSS5VK_DEBUG");
  const char* instanceExtensions[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
  if (validation_ || debug) { instanceInfo.enabledExtensionCount = 1; instanceInfo.ppEnabledExtensionNames = instanceExtensions; }
  if (validation_) {
    uint32_t count = 0;
    VK_CHECK(vkEnumerateInstanceLayerProperties(&count, nullptr));
    std::vector<VkLayerProperties> layers(count);
    VK_CHECK(vkEnumerateInstanceLayerProperties(&count, layers.data()));
    auto layer = std::find_if(layers.begin(), layers.end(), [](const VkLayerProperties& l) { return !strcmp(l.layerName, kValidationLayer); });
    if (layer == layers.end())
      throw std::runtime_error(std::string("DLSS5VK_VALIDATION=1 but ") + kValidationLayer +
                               " is not installed (a Vulkan SDK, or VK_LAYER_PATH at a Vulkan-ValidationLayers build)");
    instanceInfo.enabledLayerCount = 1;
    instanceInfo.ppEnabledLayerNames = &kValidationLayer;
    fprintf(stderr, "validation: %s enabled (API %u.%u.%u)\n", kValidationLayer, VK_API_VERSION_MAJOR(layer->specVersion),
            VK_API_VERSION_MINOR(layer->specVersion), VK_API_VERSION_PATCH(layer->specVersion));
  }
  VK_CHECK(vkCreateInstance(&instanceInfo, nullptr, &instance_));
  volkLoadInstance(instance_);
  if (validation_ || debug) {
    VkDebugUtilsMessengerCreateInfoEXT messengerInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    messengerInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    if (debug) messengerInfo.messageSeverity |= VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
    messengerInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    messengerInfo.pfnUserCallback = onDebugMessage;
    VK_CHECK(vkCreateDebugUtilsMessengerEXT(instance_, &messengerInfo, nullptr, &messenger_));
  }

  uint32_t count = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, nullptr));
  if (!count) throw std::runtime_error("no Vulkan physical devices");
  std::vector<VkPhysicalDevice> devices(count);
  VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, devices.data()));
  // Prefer a discrete device exposing FP8 cooperative matrices.
  for (VkPhysicalDevice candidate : devices) {
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extensionCount, extensions.data());
    bool coop = false, fp8 = false;
    for (const auto& extension : extensions) {
      if (!strcmp(extension.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) coop = true;
      if (!strcmp(extension.extensionName, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME)) fp8 = true;
      if (getenv("DLSS5VK_LIST_EXTENSIONS") && (strstr(extension.extensionName, "cooperative") || strstr(extension.extensionName, "NV_")))
        printf("  device extension: %s (v%u)\n", extension.extensionName, extension.specVersion);
    }
    if (coop && fp8) { physical_ = candidate; break; }
  }
  if (!physical_) throw std::runtime_error("no device with VK_KHR_cooperative_matrix + VK_EXT_shader_float8");
  if (getenv("DLSS5VK_LIST_EXTENSIONS")) {
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV cm2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &cm2;
    vkGetPhysicalDeviceFeatures2(physical_, &f2);
    printf("coopmat2 features: workgroupScope %u flexibleDimensions %u reductions %u conversions %u perElement %u tensorAddressing %u blockLoads %u\n",
           cm2.cooperativeMatrixWorkgroupScope, cm2.cooperativeMatrixFlexibleDimensions, cm2.cooperativeMatrixReductions,
           cm2.cooperativeMatrixConversions, cm2.cooperativeMatrixPerElementOperations, cm2.cooperativeMatrixTensorAddressing,
           cm2.cooperativeMatrixBlockLoads);
    VkPhysicalDeviceCooperativeMatrix2PropertiesNV p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_PROPERTIES_NV};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &p2;
    vkGetPhysicalDeviceProperties2(physical_, &props2);
    printf("coopmat2 properties: maxWorkgroupSize %u maxFlexibleDimension %u reservedShared %u\n",
           p2.cooperativeMatrixWorkgroupScopeMaxWorkgroupSize, p2.cooperativeMatrixFlexibleDimensionsMaxDimension,
           p2.cooperativeMatrixWorkgroupScopeReservedSharedMemory);
    auto getFlex = (PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV)vkGetInstanceProcAddr(
        instance_, "vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV");
    if (getFlex) {
      uint32_t n = 0;
      getFlex(physical_, &n, nullptr);
      std::vector<VkCooperativeMatrixFlexibleDimensionsPropertiesNV> flex(n);
      for (auto& f : flex) f.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_FLEXIBLE_DIMENSIONS_PROPERTIES_NV;
      getFlex(physical_, &n, flex.data());
      for (auto& f : flex)
        printf("  flexible: gran M%u N%u K%u  A %u B %u C %u R %u  sat %u scope %u wgInvocations %u\n", f.MGranularity, f.NGranularity,
               f.KGranularity, f.AType, f.BType, f.CType, f.ResultType, f.saturatingAccumulation, f.scope, f.workgroupInvocations);
    }
  }

  VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceShaderSMBuiltinsPropertiesNV smBuiltins{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV};
  properties.pNext = &subgroup;
  subgroup.pNext = &smBuiltins;
  vkGetPhysicalDeviceProperties2(physical_, &properties);
  smCount_ = smBuiltins.shaderSMCount;
  deviceName_ = properties.properties.deviceName;
  timestampPeriod_ = properties.properties.limits.timestampPeriod;
  maxSharedMemory_ = properties.properties.limits.maxComputeSharedMemorySize;
  if (subgroup.subgroupSize != 32)
    fprintf(stderr, "warning: subgroup size %u (cooperative kernels assume 32)\n", subgroup.subgroupSize);
  vkGetPhysicalDeviceMemoryProperties(physical_, &memoryProperties_);

  uint32_t familyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_, &familyCount, nullptr);
  std::vector<VkQueueFamilyProperties> families(familyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_, &familyCount, families.data());
  queueFamily_ = UINT32_MAX;
  for (uint32_t index = 0; index < familyCount; ++index) {
    if ((families[index].queueFlags & VK_QUEUE_COMPUTE_BIT) && families[index].timestampValidBits) {
      // Prefer a family that also has graphics (the "main" queue) for widest support.
      if (queueFamily_ == UINT32_MAX || (families[index].queueFlags & VK_QUEUE_GRAPHICS_BIT)) queueFamily_ = index;
    }
  }
  if (queueFamily_ == UINT32_MAX) throw std::runtime_error("no compute queue with timestamps");

  float priority = 1.0f;
  VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queueInfo.queueFamilyIndex = queueFamily_;
  queueInfo.queueCount = 1;
  queueInfo.pQueuePriorities = &priority;

  DeviceRequirements req;
  VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  deviceInfo.pNext = &req.features;
  deviceInfo.queueCreateInfoCount = 1;
  deviceInfo.pQueueCreateInfos = &queueInfo;
  deviceInfo.enabledExtensionCount = (uint32_t)req.extensions.size();
  deviceInfo.ppEnabledExtensionNames = req.extensions.data();
  VK_CHECK(vkCreateDevice(physical_, &deviceInfo, nullptr, &device_));
  volkLoadDevice(device_);
  initCommon();
}

Context::Context(VkInstance instance, VkPhysicalDevice physical, VkDevice device, uint32_t queueFamily, uint32_t queueIndex) {
  if (volkInitialize() != VK_SUCCESS) throw std::runtime_error("vulkan-1.dll unavailable");
  instance_ = instance; physical_ = physical; device_ = device; queueFamily_ = queueFamily; queueIndex_ = queueIndex; owned_ = false;
  volkLoadInstance(instance_);
  volkLoadDevice(device_);
  // A throw destroys what was made so far on the application's device (no destructor runs for it).
  try {
    adopt();
  } catch (...) {
    release();
    throw;
  }
}

void Context::adopt() {
  VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceShaderSMBuiltinsPropertiesNV smBuiltins{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV};
  properties.pNext = &subgroup;
  subgroup.pNext = &smBuiltins;
  vkGetPhysicalDeviceProperties2(physical_, &properties);
  smCount_ = smBuiltins.shaderSMCount;
  deviceName_ = properties.properties.deviceName;
  timestampPeriod_ = properties.properties.limits.timestampPeriod;
  maxSharedMemory_ = properties.properties.limits.maxComputeSharedMemorySize;
  vkGetPhysicalDeviceMemoryProperties(physical_, &memoryProperties_);
  initCommon();
}

DeviceRequirements::DeviceRequirements() {
  f11.storageBuffer16BitAccess = VK_TRUE;
  f11.uniformAndStorageBuffer16BitAccess = VK_TRUE;
  f12.pNext = &f11;
  f12.storageBuffer8BitAccess = VK_TRUE;
  f12.uniformAndStorageBuffer8BitAccess = VK_TRUE;
  f12.shaderFloat16 = VK_TRUE;
  f12.shaderInt8 = VK_TRUE;
  f12.vulkanMemoryModel = VK_TRUE;
  f12.vulkanMemoryModelDeviceScope = VK_TRUE;
  f12.hostQueryReset = VK_TRUE;
  f12.bufferDeviceAddress = VK_TRUE;
  f13.pNext = &f12;
  f13.subgroupSizeControl = VK_TRUE;
  f13.computeFullSubgroups = VK_TRUE;
  f13.synchronization2 = VK_TRUE;
  f13.maintenance4 = VK_TRUE;   // LocalSizeId: gemm_fp8.comp takes its workgroup size from a specialization constant
  coop.pNext = &f13;
  coop.cooperativeMatrix = VK_TRUE;
  coop2.pNext = &coop;
  coop2.cooperativeMatrixWorkgroupScope = VK_TRUE;
  coop2.cooperativeMatrixFlexibleDimensions = VK_TRUE;
  coop2.cooperativeMatrixReductions = VK_TRUE;
  coop2.cooperativeMatrixConversions = VK_TRUE;
  coop2.cooperativeMatrixPerElementOperations = VK_TRUE;
  coop2.cooperativeMatrixTensorAddressing = VK_TRUE;
  coop2.cooperativeMatrixBlockLoads = VK_TRUE;
  fp8.pNext = &coop2;
  fp8.shaderFloat8 = VK_TRUE;
  fp8.shaderFloat8CooperativeMatrix = VK_TRUE;
  features.pNext = &fp8;
  features.features.shaderInt16 = VK_TRUE;
  features.features.shaderInt64 = VK_TRUE;
  // robustBufferAccess stays off (15% cost, masks bugs); experiment shaders must check their own ranges.
  features.features.robustBufferAccess = VK_FALSE;

  VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR& executableFeatures = executable;
  executableFeatures.pipelineExecutableInfo = VK_TRUE;
  executableFeatures.pNext = features.pNext;
  features.pNext = &executableFeatures;
  // Shader clocks (clockARB / clockRealtimeEXT).
  VkPhysicalDeviceShaderClockFeaturesKHR& clockFeatures = clock;
  clockFeatures.shaderSubgroupClock = VK_TRUE;
  clockFeatures.shaderDeviceClock = VK_TRUE;
  clockFeatures.pNext = features.pNext;
  features.pNext = &clockFeatures;
  VkPhysicalDeviceShaderSMBuiltinsFeaturesNV& smFeatures = sm;
  smFeatures.shaderSMBuiltins = VK_TRUE;
  smFeatures.pNext = features.pNext;
  features.pNext = &smFeatures;
  extensions = {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME,
                              VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME, VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME,
                              VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME, VK_KHR_SHADER_CLOCK_EXTENSION_NAME,
                              VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME};
}

void Context::initCommon() {
  vkGetDeviceQueue(device_, queueFamily_, queueIndex_, &queue_);

  VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = queueFamily_;
  VK_CHECK(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_));

  VkDescriptorSetLayoutBinding bindings[kGenericBindings];
  for (uint32_t index = 0; index < kGenericBindings; ++index) {
    bindings[index] = {index, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  }
  VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layoutInfo.bindingCount = kGenericBindings;
  layoutInfo.pBindings = bindings;
  VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &setLayout_));

  VkPushConstantRange pushRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushConstantBytes};
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &setLayout_;
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges = &pushRange;
  VK_CHECK(vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_));

  VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8192 * kGenericBindings};
  VkDescriptorPoolCreateInfo descriptorPoolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  descriptorPoolInfo.maxSets = 8192;
  descriptorPoolInfo.poolSizeCount = 1;
  descriptorPoolInfo.pPoolSizes = &poolSize;
  for (VkDescriptorPool& pool : descriptorPools_) VK_CHECK(vkCreateDescriptorPool(device_, &descriptorPoolInfo, nullptr, &pool));
  descriptorPool_ = descriptorPools_[0];

  dummy_ = createBuffer(256, false, "dummy binding");
  fillZero(dummy_);
  staging_ = createBuffer(kStagingBytes, true, "staging");
}

Context::~Context() { release(); }

void Context::release() {
  if (!device_) return;
  vkDeviceWaitIdle(device_);
  for (VkShaderModule module : modules_) vkDestroyShaderModule(device_, module, nullptr);
  destroyBuffer(dummy_);
  destroyBuffer(staging_);
  for (VkDescriptorPool pool : descriptorPools_) vkDestroyDescriptorPool(device_, pool, nullptr);
  vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
  vkDestroyDescriptorSetLayout(device_, setLayout_, nullptr);
  vkDestroyCommandPool(device_, commandPool_, nullptr);
  if (owned_) {
    vkDestroyDevice(device_, nullptr);
    if (messenger_) vkDestroyDebugUtilsMessengerEXT(instance_, messenger_, nullptr);
    vkDestroyInstance(instance_, nullptr);
  }
  device_ = VK_NULL_HANDLE;
}

uint32_t Context::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required) {
  for (uint32_t index = 0; index < memoryProperties_.memoryTypeCount; ++index) {
    if ((typeBits & (1u << index)) &&
        (memoryProperties_.memoryTypes[index].propertyFlags & required) == required)
      return index;
  }
  throw std::runtime_error("no suitable memory type");
}

Buffer Context::createBuffer(VkDeviceSize size, bool hostVisible, const char* label, VkBufferUsageFlags extra) {
  Buffer result;
  result.size = std::max<VkDeviceSize>(size, 16);
  result.hostVisible = hostVisible;
  result.label = label;
  VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = result.size;
  info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | extra;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(device_, &info, nullptr, &result.buffer));
  // A failure past here (no memory type, an allocation the device refuses) destroys the buffer.
  try {
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device_, result.buffer, &requirements);
    VkMemoryAllocateFlagsInfo allocateFlags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    allocateFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocateInfo.pNext = &allocateFlags;
    allocateInfo.allocationSize = requirements.size;
    allocateInfo.memoryTypeIndex = findMemoryType(
        requirements.memoryTypeBits,
        hostVisible ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                    : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory memory = VK_NULL_HANDLE;   // written only on success
    VK_CHECK(vkAllocateMemory(device_, &allocateInfo, nullptr, &memory));
    result.memory = memory;
    VK_CHECK(vkBindBufferMemory(device_, result.buffer, result.memory, 0));
    if (hostVisible) VK_CHECK(vkMapMemory(device_, result.memory, 0, VK_WHOLE_SIZE, 0, &result.mapped));
  } catch (...) {
    destroyBuffer(result);
    throw;
  }
  return result;
}

void Context::destroyBuffer(Buffer& buffer) {
  if (buffer.mapped) vkUnmapMemory(device_, buffer.memory);
  if (buffer.buffer) vkDestroyBuffer(device_, buffer.buffer, nullptr);
  if (buffer.memory) vkFreeMemory(device_, buffer.memory, nullptr);
  buffer = Buffer{};
}

void Context::upload(const Buffer& target, const void* data, VkDeviceSize size, VkDeviceSize offset) {
  if (offset + size > target.size) throw std::runtime_error(std::string("upload overflows ") + target.label);
  const uint8_t* bytes = static_cast<const uint8_t*>(data);
  VkDeviceSize done = 0;
  while (done < size) {
    VkDeviceSize chunk = std::min(size - done, staging_.size);
    memcpy(staging_.mapped, bytes + done, chunk);
    VkCommandBuffer commands = beginCommands();
    VkBufferCopy region{0, offset + done, chunk};
    vkCmdCopyBuffer(commands, staging_.buffer, target.buffer, 1, &region);
    endAndSubmit(commands, true);
    done += chunk;
  }
}

void Context::fillZero(const Buffer& target) {
  VkCommandBuffer commands = beginCommands();
  vkCmdFillBuffer(commands, target.buffer, 0, VK_WHOLE_SIZE, 0);
  endAndSubmit(commands, true);
}

std::vector<uint8_t> Context::download(const Buffer& source, VkDeviceSize size, VkDeviceSize offset) {
  if (offset + size > source.size) throw std::runtime_error(std::string("download overflows ") + source.label);
  std::vector<uint8_t> result(size);
  VkDeviceSize done = 0;
  while (done < size) {
    VkDeviceSize chunk = std::min(size - done, staging_.size);
    VkCommandBuffer commands = beginCommands();
    VkBufferCopy region{offset + done, 0, chunk};
    vkCmdCopyBuffer(commands, source.buffer, staging_.buffer, 1, &region);
    endAndSubmit(commands, true);
    memcpy(result.data() + done, staging_.mapped, chunk);
    done += chunk;
  }
  return result;
}

VkShaderModule Context::loadShaderModule(const std::string& spvPath) {
  std::ifstream file(spvPath, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("missing shader " + spvPath);
  std::streamsize size = file.tellg();
  file.seekg(0);
  std::vector<uint32_t> words((size + 3) / 4);
  file.read(reinterpret_cast<char*>(words.data()), size);
  VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  info.codeSize = size;
  info.pCode = words.data();
  VkShaderModule module;
  VK_CHECK(vkCreateShaderModule(device_, &info, nullptr, &module));
  modules_.push_back(module);
  return module;
}

Pipeline Context::createComputePipeline(VkShaderModule module, const SpecConstants& constants,
                                        const char* label, uint32_t requiredSubgroupSize) {
  VkSpecializationInfo specialization{};
  specialization.mapEntryCount = (uint32_t)constants.entries.size();
  specialization.pMapEntries = constants.entries.data();
  specialization.dataSize = constants.data.size() * 4;
  specialization.pData = constants.data.data();
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupInfo{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
  subgroupInfo.requiredSubgroupSize = requiredSubgroupSize;
  VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.pNext = requiredSubgroupSize ? &subgroupInfo : nullptr;
  stage.flags = requiredSubgroupSize ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT : 0;
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = module;
  stage.pName = "main";
  stage.pSpecializationInfo = constants.entries.empty() ? nullptr : &specialization;
  VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  info.stage = stage;
  info.layout = pipelineLayout_;
  if (captureStatistics_)
    info.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
  Pipeline result;
  result.label = label;
  VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &result.pipeline));
  return result;
}

std::string Context::pipelineStatistics(const Pipeline& pipeline, bool includeInternal) {
  std::string report;
  VkPipelineInfoKHR pipelineInfo{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
  pipelineInfo.pipeline = pipeline.pipeline;
  uint32_t executableCount = 0;
  if (vkGetPipelineExecutablePropertiesKHR(device_, &pipelineInfo, &executableCount, nullptr) != VK_SUCCESS) return "n/a";
  std::vector<VkPipelineExecutablePropertiesKHR> executables(executableCount,
                                                             {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
  vkGetPipelineExecutablePropertiesKHR(device_, &pipelineInfo, &executableCount, executables.data());
  for (uint32_t e = 0; e < executableCount; ++e) {
    report += std::string(executables[e].name) + " (" + executables[e].description + ") subgroup " +
              std::to_string(executables[e].subgroupSize) + "\n";
    VkPipelineExecutableInfoKHR executableInfo{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
    executableInfo.pipeline = pipeline.pipeline;
    executableInfo.executableIndex = e;
    uint32_t statisticCount = 0;
    vkGetPipelineExecutableStatisticsKHR(device_, &executableInfo, &statisticCount, nullptr);
    std::vector<VkPipelineExecutableStatisticKHR> statistics(statisticCount,
                                                             {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
    vkGetPipelineExecutableStatisticsKHR(device_, &executableInfo, &statisticCount, statistics.data());
    for (const auto& statistic : statistics) {
      report += "  " + std::string(statistic.name) + " = ";
      switch (statistic.format) {
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: report += statistic.value.b32 ? "true" : "false"; break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: report += std::to_string(statistic.value.i64); break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: report += std::to_string(statistic.value.u64); break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR: report += std::to_string(statistic.value.f64); break;
        default: break;
      }
      report += "  (" + std::string(statistic.description) + ")\n";
    }
    if (includeInternal) {
      uint32_t irCount = 0;
      vkGetPipelineExecutableInternalRepresentationsKHR(device_, &executableInfo, &irCount, nullptr);
      std::vector<VkPipelineExecutableInternalRepresentationKHR> irs(
          irCount, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR});
      vkGetPipelineExecutableInternalRepresentationsKHR(device_, &executableInfo, &irCount, irs.data());
      std::vector<std::vector<char>> storage(irCount);
      for (uint32_t i = 0; i < irCount; ++i) { storage[i].resize(irs[i].dataSize + 1); irs[i].pData = storage[i].data(); }
      vkGetPipelineExecutableInternalRepresentationsKHR(device_, &executableInfo, &irCount, irs.data());
      for (uint32_t i = 0; i < irCount; ++i) {
        report += "  --- " + std::string(irs[i].name) + " (" + irs[i].description + ", " + std::to_string(irs[i].dataSize) +
                  " bytes, text=" + (irs[i].isText ? "yes" : "no") + ")\n";
        if (irs[i].isText) report += std::string(storage[i].data(), irs[i].dataSize) + "\n";
      }
    }
  }
  return report;
}

void Context::destroyPipeline(Pipeline& pipeline) {
  if (pipeline.pipeline) vkDestroyPipeline(device_, pipeline.pipeline, nullptr);
  pipeline = Pipeline{};
}

VkDescriptorSet Context::allocateSet(const Buffer* const bindings[kGenericBindings],
                                     const VkDeviceSize offsets[kGenericBindings],
                                     const VkDeviceSize ranges[kGenericBindings]) {
  VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocateInfo.descriptorPool = descriptorPool_;
  allocateInfo.descriptorSetCount = 1;
  allocateInfo.pSetLayouts = &setLayout_;
  VkDescriptorSet set;
  VK_CHECK(vkAllocateDescriptorSets(device_, &allocateInfo, &set));
  VkDescriptorBufferInfo infos[kGenericBindings];
  VkWriteDescriptorSet writes[kGenericBindings];
  for (uint32_t index = 0; index < kGenericBindings; ++index) {
    const Buffer* buffer = bindings[index] ? bindings[index] : &dummy_;
    infos[index] = {buffer->buffer, offsets ? offsets[index] : 0,
                    ranges && ranges[index] ? ranges[index] : VK_WHOLE_SIZE};
    writes[index] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[index].dstSet = set;
    writes[index].dstBinding = index;
    writes[index].descriptorCount = 1;
    writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[index].pBufferInfo = &infos[index];
  }
  vkUpdateDescriptorSets(device_, kGenericBindings, writes, 0, nullptr);
  return set;
}

// Two pools in rotation: a frame's sets stay valid while the next frame is recorded (the caller waits for the
// frame before the previous one, as the demo's two frames in flight do).
void Context::resetDescriptorPool() { resetDescriptorPool((uint32_t)((descriptorPoolIndex_ + 1) % descriptorPools_.size())); }

void Context::resetDescriptorPool(uint32_t slot) {
  descriptorPoolIndex_ = slot % descriptorPools_.size();
  descriptorPool_ = descriptorPools_[descriptorPoolIndex_];
  VK_CHECK(vkResetDescriptorPool(device_, descriptorPool_, 0));
}

VkCommandBuffer Context::beginCommands() {
  VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocateInfo.commandPool = commandPool_;
  allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocateInfo.commandBufferCount = 1;
  VkCommandBuffer commands;
  VK_CHECK(vkAllocateCommandBuffers(device_, &allocateInfo, &commands));
  VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(commands, &beginInfo));
  return commands;
}

void Context::endAndSubmit(VkCommandBuffer commands, bool wait) {
  VK_CHECK(vkEndCommandBuffer(commands));
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &commands;
  VK_CHECK(vkQueueSubmit(queue_, 1, &submit, VK_NULL_HANDLE));
  if (wait) {
    VK_CHECK(vkQueueWaitIdle(queue_));
    vkFreeCommandBuffers(device_, commandPool_, 1, &commands);
  }
}

void Context::computeBarrier(VkCommandBuffer commands) {
  // Compute -> compute only. Including the transfer stages here made NVIDIA flush
  // caches between every dispatch (tens of microseconds per barrier once L2 is dirty).
  VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
  barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.memoryBarrierCount = 1;
  dependency.pMemoryBarriers = &barrier;
  vkCmdPipelineBarrier2(commands, &dependency);
}

// Captures and uploads only.
void Context::transferBarrier(VkCommandBuffer commands) {
  VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
  barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                          VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.memoryBarrierCount = 1;
  dependency.pMemoryBarriers = &barrier;
  vkCmdPipelineBarrier2(commands, &dependency);
}

VkQueryPool Context::createTimestampPool(uint32_t count) {
  VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  info.queryCount = count;
  VkQueryPool pool;
  VK_CHECK(vkCreateQueryPool(device_, &info, nullptr, &pool));
  vkResetQueryPool(device_, pool, 0, count);
  return pool;
}

std::vector<double> Context::readTimestampsMs(VkQueryPool pool, uint32_t count) {
  std::vector<uint64_t> ticks(count);
  VK_CHECK(vkGetQueryPoolResults(device_, pool, 0, count, ticks.size() * 8, ticks.data(), 8,
                                 VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
  std::vector<double> result(count);
  for (uint32_t index = 0; index < count; ++index) result[index] = ticks[index] * (double)timestampPeriod_ * 1e-6;
  return result;
}

}  // namespace vk

namespace vk {

VkDeviceAddress Context::deviceAddress(const Buffer& buffer) const {
  VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  info.buffer = buffer.buffer;
  return vkGetBufferDeviceAddress(device_, &info);
}

VkCudaModuleNV Context::createCudaModule(const std::string& ptx) {
  VkCudaModuleCreateInfoNV info{VK_STRUCTURE_TYPE_CUDA_MODULE_CREATE_INFO_NV};
  info.dataSize = ptx.size() + 1;
  info.pData = ptx.c_str();
  VkCudaModuleNV module = VK_NULL_HANDLE;
  VK_CHECK(vkCreateCudaModuleNV(device_, &info, nullptr, &module));
  return module;
}

VkCudaFunctionNV Context::createCudaFunction(VkCudaModuleNV module, const char* name) {
  VkCudaFunctionCreateInfoNV info{VK_STRUCTURE_TYPE_CUDA_FUNCTION_CREATE_INFO_NV};
  info.module = module;
  info.pName = name;
  VkCudaFunctionNV function = VK_NULL_HANDLE;
  VK_CHECK(vkCreateCudaFunctionNV(device_, &info, nullptr, &function));
  return function;
}

void Context::destroyCudaFunction(VkCudaFunctionNV function) {
  if (function) vkDestroyCudaFunctionNV(device_, function, nullptr);
}

void Context::destroyCudaModule(VkCudaModuleNV module) {
  if (module) vkDestroyCudaModuleNV(device_, module, nullptr);
}

void Context::cudaLaunch(VkCommandBuffer commands, VkCudaFunctionNV function, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                         uint32_t blockX, uint32_t sharedBytes, const void* const* params, size_t paramCount) {
  VkCudaLaunchInfoNV info{VK_STRUCTURE_TYPE_CUDA_LAUNCH_INFO_NV};
  info.function = function;
  info.gridDimX = gridX; info.gridDimY = gridY; info.gridDimZ = gridZ;
  info.blockDimX = blockX; info.blockDimY = 1; info.blockDimZ = 1;
  info.sharedMemBytes = sharedBytes;
  info.paramCount = paramCount;
  info.pParams = params;
  vkCmdCudaLaunchKernelNV(commands, &info);
}

}  // namespace vk
