#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <set>
#include <algorithm>
#include <functional>
#define NOMINMAX  
#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include "variables.hpp"

#define CL_HPP_ENABLE_EXCEPTIONS  

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/opencl.hpp>
#endif

struct QueueFamilyIndices
{
    int graphicsFamily = -1;
    int presentFamily  = -1;

    bool isComplete()
    {
        return graphicsFamily >= 0 && presentFamily >= 0;
    }
    
};

struct SwapchainSupportDetails
{
    VkSurfaceCapabilitiesKHR        capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   presentModes;
};

class Simulation
{
    public:
        void init()
        {
            initWindow();
            initVulkan();
        }

        void loop(std::function<void()> simulateStep)
        {
            mainLoop(simulateStep);
            cleanup();
        }
        HANDLE getSharedMemoryHandle()
        {
            VkMemoryGetWin32HandleInfoKHR handleInfo{};
            handleInfo.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
            handleInfo.memory     = vertexBufferMemory;
            handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

            auto vkGetMemoryWin32HandleKHR = (PFN_vkGetMemoryWin32HandleKHR)
                vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");

            if (!vkGetMemoryWin32HandleKHR)
            {
                throw std::runtime_error("vkGetMemoryWin32HandleKHR not available!");
            }

            HANDLE handle;
            vkGetMemoryWin32HandleKHR(device, &handleInfo, &handle);
            sharedHandle = handle;
            return handle;
        }

        void* getMappedVertexBuffer()
        {
            VkDeviceSize bufferSize = sizeof(float) * 3 * NUM_BODIES;
            vkMapMemory(device, vertexBufferMemory, 0, bufferSize, 0, &mappedVertexBuffer);
            return mappedVertexBuffer;
        }

        float camZoom   = 1.0f;
        float camX      = 0.0f;
        float camY      = 0.0f;
        bool  dragging  = false;
        double dragStartX = 0.0;
        double dragStartY = 0.0;
        float  dragCamStartX = 0.0f;
        float  dragCamStartY = 0.0f;
    private:
        GLFWwindow* window;
        VkInstance instance;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkSurfaceKHR surface;
        VkDevice device;                    
        VkQueue graphicsQueue;
        VkQueue presentQueue;
        VkSwapchainKHR swapchain;                
        std::vector<VkImage> swapchainImages;
        VkFormat swapchainImageFormat;
        VkExtent2D swapchainExtent;
        std::vector<VkImageView> swapchainImageViews;
        VkRenderPass renderPass;
        VkPipelineLayout pipelineLayout;
        VkPipeline graphicsPipeline;
        std::vector<VkFramebuffer> swapchainFramebuffers;
        VkBuffer vertexBuffer;
        VkDeviceMemory vertexBufferMemory;
        VkCommandPool commandPool;
        std::vector<VkCommandBuffer> commandBuffers;
        VkSemaphore imageAvailableSemaphore;
        VkSemaphore renderFinishedSemaphore;
        VkFence inFlightFence;
        HANDLE sharedHandle = nullptr;
        void* mappedVertexBuffer = nullptr;

        void initWindow() 
        {
            if (!glfwInit()) 
            {
                exit(1);
            }

            GLFWmonitor* monitor = glfwGetPrimaryMonitor();
            if (!monitor) 
            {
                exit(1);
            }

            const GLFWvidmode* mode = glfwGetVideoMode(monitor);
            if (!mode) 
            {
                exit(1);
            }

            std::cout << mode->width <<  " " << mode->height << std::endl;

            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            //glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

            window = glfwCreateWindow(mode->width, mode->height, "Window", nullptr, nullptr);
            glfwMakeContextCurrent(window);

            glfwSetWindowUserPointer(window, this);

            // Q / Escape to quit
            glfwSetKeyCallback(window, [](GLFWwindow* w, int key, int, int action, int)
            {
                if (action == GLFW_PRESS && (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE))
                    glfwSetWindowShouldClose(w, GLFW_TRUE);
            });

            // Scroll to zoom
            glfwSetScrollCallback(window, [](GLFWwindow* w, double, double yOffset)
            {
                auto* sim = reinterpret_cast<Simulation*>(glfwGetWindowUserPointer(w));
                float factor = (yOffset > 0) ? 1.1f : 0.9f;
                sim->camZoom *= factor;
                sim->camZoom  = std::max(0.0001f, std::min(sim->camZoom, 100000.0f));
            });

            glfwSetMouseButtonCallback(window, [](GLFWwindow* w, int button, int action, int)
            {
                auto* sim = reinterpret_cast<Simulation*>(glfwGetWindowUserPointer(w));
                if (button == GLFW_MOUSE_BUTTON_LEFT)
                {
                    if (action == GLFW_PRESS)
                    {
                        sim->dragging = true;
                        glfwGetCursorPos(w, &sim->dragStartX, &sim->dragStartY);
                        sim->dragCamStartX = sim->camX;
                        sim->dragCamStartY = sim->camY;
                    }
                    else if (action == GLFW_RELEASE)
                    {
                        sim->dragging = false;
                    }
                }
            });

            glfwSetCursorPosCallback(window, [](GLFWwindow* w, double xpos, double ypos)
            {
                auto* sim = reinterpret_cast<Simulation*>(glfwGetWindowUserPointer(w));
                if (!sim->dragging) return;

                double dx = xpos - sim->dragStartX;
                double dy = ypos - sim->dragStartY;

                // Convert pixel delta to simulation space
                // 6000 / zoom maps screen fraction back to world units
                int winW, winH;
                glfwGetWindowSize(w, &winW, &winH);

                sim->camX = sim->dragCamStartX - (float)(dx / winW) * SPAWN_RANGE / sim->camZoom;
                sim->camY = sim->dragCamStartY - (float)(dy / winH) * SPAWN_RANGE / sim->camZoom;
            });
        }

        void initVulkan() 
        {
            createInstance();
            createSurface(); 
            pickPhysicalDevice();
            createLogicalDevice();
            createSwapchain();
            createImageViews();
            createRenderPass();
            createGraphicsPipeline();
            createFramebuffers();
            createVertexBuffer();
            createCommandPool();
            createSyncObjects(); 
        }

        void createInstance() 
        {
            VkApplicationInfo appInfo{};
            appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            appInfo.pApplicationName = "Hello Triangle";
            appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
            appInfo.pEngineName = "No Engine";
            appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
            appInfo.apiVersion = VK_API_VERSION_1_0;
            
            VkInstanceCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            createInfo.pApplicationInfo = &appInfo;
            
            uint32_t glfwExtensionCount = 0;
            const char** glfwExtensions;

            glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
            
            std::vector<const char*> instanceExtensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
            instanceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
            
            createInfo.enabledExtensionCount   = static_cast<uint32_t>(instanceExtensions.size());
            createInfo.ppEnabledExtensionNames = instanceExtensions.data();
            
            createInfo.enabledLayerCount = 0;
            
            VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
            if (result != VK_SUCCESS) {
                std::cerr << "Vulkan Error Code: " << result << std::endl;
                // Common codes: 
                // -1 (OUT_OF_HOST_MEMORY)
                // -3 (INITIALIZATION_FAILED)
                // -7 (EXTENSION_NOT_PRESENT)
                // -9 (FEATURE_NOT_PRESENT)
                throw std::runtime_error("failed to create instance!");
            }
            
        }

        void pickPhysicalDevice()
        {
            uint32_t deviceCount = 0;
            vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

            if (deviceCount == 0)
            {
                throw std::runtime_error("No GPUs with Vulkan support found!");
            }

            std::vector<VkPhysicalDevice> devices(deviceCount);
            vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

            for (const auto& device : devices)
            {
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(device, &props);

                std::cout << "Found GPU: " << props.deviceName << std::endl;

                if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
                {
                    physicalDevice = device;
                    std::cout << "Selected: " << props.deviceName << std::endl;
                    break;
                }
            }

            if (physicalDevice == VK_NULL_HANDLE)
            {
                physicalDevice = devices[0];

                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(physicalDevice, &props);
                std::cout << "No discrete GPU found, falling back to: " << props.deviceName << std::endl;
            }
        }

        void createSurface()
        {
            if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create window surface!");
            }

            std::cout << "Surface created successfully" << std::endl;
        }

        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
        {
            QueueFamilyIndices indices;

            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

            std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

            for (int i = 0; i < queueFamilies.size(); i++)
            {
                // Check for graphics support
                if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                {
                    indices.graphicsFamily = i;
                }

                // Check for present support on this surface
                VkBool32 presentSupport = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
                if (presentSupport)
                {
                    indices.presentFamily = i;
                }

                if (indices.isComplete())
                {
                    break;
                }
            }

            return indices;
        }

        void createLogicalDevice()
        {
            QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

            const std::vector<const char*> deviceExtensions = {
                VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
                VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME  // Windows specific
            };
            
            // Build a set of unique queue families needed
            // (graphics and present are often the same family)
            std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
            std::set<int> uniqueQueueFamilies = {
                indices.graphicsFamily,
                indices.presentFamily
            };

            float queuePriority = 1.0f;

            for (int queueFamily : uniqueQueueFamilies)
            {
                VkDeviceQueueCreateInfo queueCreateInfo{};
                queueCreateInfo.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                queueCreateInfo.queueFamilyIndex = queueFamily;
                queueCreateInfo.queueCount       = 1;
                queueCreateInfo.pQueuePriorities = &queuePriority;
                queueCreateInfos.push_back(queueCreateInfo);
            }

            // No special features needed yet
            VkPhysicalDeviceFeatures deviceFeatures{};

            VkDeviceCreateInfo createInfo{};
            createInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            createInfo.queueCreateInfoCount    = static_cast<uint32_t>(queueCreateInfos.size());
            createInfo.pQueueCreateInfos       = queueCreateInfos.data();
            createInfo.pEnabledFeatures        = &deviceFeatures;
            createInfo.enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size());
            createInfo.ppEnabledExtensionNames = deviceExtensions.data();
            createInfo.enabledLayerCount       = 0;

            if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
                throw std::runtime_error("Failed to create logical device!");

            // Retrieve the actual queue handles
            vkGetDeviceQueue(device, indices.graphicsFamily, 0, &graphicsQueue);
            vkGetDeviceQueue(device, indices.presentFamily,  0, &presentQueue);

            std::cout << "Logical device created successfully" << std::endl;
            std::cout << "Graphics family: " << indices.graphicsFamily << std::endl;
            std::cout << "Present family:  " << indices.presentFamily  << std::endl;
        }

        SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice device)
        {
            SwapchainSupportDetails details;

            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

            uint32_t formatCount = 0;
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
            if (formatCount != 0)
            {
                details.formats.resize(formatCount);
                vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
            }

            uint32_t presentModeCount = 0;
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
            if (presentModeCount != 0)
            {
                details.presentModes.resize(presentModeCount);
                vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
            }

            return details;
        }

        // Pick the color format — prefer 32bit RGBA in sRGB color space
        VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
        {
            for (const auto& format : availableFormats)
            {
                if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
                    format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                    {
                        return format;
                    }
            }

            // Fall back to whatever is first
            return availableFormats[0];
        }

        // Pick the present mode — prefer Mailbox (triple buffering), fall back to FIFO (vsync)
        VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
        {
            for (const auto& mode : availablePresentModes)
            {
                if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
                {
                    std::cout << "Present mode: Mailbox (triple buffering)" << std::endl;
                    return mode;
                }
            }

            std::cout << "Present mode: FIFO (vsync)" << std::endl;
            return VK_PRESENT_MODE_FIFO_KHR;
        }

        // Pick the resolution of the swapchain images
        VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
        {
            if (capabilities.currentExtent.width != UINT32_MAX)
                return capabilities.currentExtent;

            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width  = std::max(capabilities.minImageExtent.width,
                                std::min(capabilities.maxImageExtent.width,  actualExtent.width));
            actualExtent.height = std::max(capabilities.minImageExtent.height,
                                std::min(capabilities.maxImageExtent.height, actualExtent.height));

            return actualExtent;
        }

        void createSwapchain()
        {
            SwapchainSupportDetails swapchainSupport = querySwapchainSupport(physicalDevice);

            VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapchainSupport.formats);
            VkPresentModeKHR   presentMode   = chooseSwapPresentMode(swapchainSupport.presentModes);
            VkExtent2D         extent        = chooseSwapExtent(swapchainSupport.capabilities);

            // Request one more image than minimum to avoid waiting on driver
            uint32_t imageCount = swapchainSupport.capabilities.minImageCount + 1;

            // Don't exceed the maximum (0 means no maximum)
            if (swapchainSupport.capabilities.maxImageCount > 0 &&
                imageCount > swapchainSupport.capabilities.maxImageCount)
                imageCount = swapchainSupport.capabilities.maxImageCount;

            VkSwapchainCreateInfoKHR createInfo{};
            createInfo.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
            createInfo.surface          = surface;
            createInfo.minImageCount    = imageCount;
            createInfo.imageFormat      = surfaceFormat.format;
            createInfo.imageColorSpace  = surfaceFormat.colorSpace;
            createInfo.imageExtent      = extent;
            createInfo.imageArrayLayers = 1;
            createInfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

            QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
            uint32_t queueFamilyIndices[] = {
                (uint32_t)indices.graphicsFamily,
                (uint32_t)indices.presentFamily
            };

            if (indices.graphicsFamily != indices.presentFamily)
            {
                // Images shared across queue families
                createInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
                createInfo.queueFamilyIndexCount = 2;
                createInfo.pQueueFamilyIndices   = queueFamilyIndices;
            }
            else
            {
                // Same queue family — exclusive is faster
                createInfo.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
                createInfo.queueFamilyIndexCount = 0;
                createInfo.pQueueFamilyIndices   = nullptr;
            }

            createInfo.preTransform   = swapchainSupport.capabilities.currentTransform;
            createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
            createInfo.presentMode    = presentMode;
            createInfo.clipped        = VK_TRUE;
            createInfo.oldSwapchain   = VK_NULL_HANDLE;

            if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create swapchain!");
            }

            // Retrieve the swapchain image handles
            vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
            swapchainImages.resize(imageCount);
            vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());

            swapchainImageFormat = surfaceFormat.format;
            swapchainExtent      = extent;

            std::cout << "Swapchain created: " << imageCount << " images, "
                    << extent.width << "x" << extent.height << std::endl;
        }

        void createImageViews()
        {
            swapchainImageViews.resize(swapchainImages.size());

            for (int i = 0; i < swapchainImages.size(); i++)
            {
                VkImageViewCreateInfo createInfo{};
                createInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                createInfo.image    = swapchainImages[i];
                createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                createInfo.format   = swapchainImageFormat;

                // No swizzling — keep RGBA as is
                createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

                // This is a color image, no mipmaps, no layers
                createInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
                createInfo.subresourceRange.baseMipLevel   = 0;
                createInfo.subresourceRange.levelCount     = 1;
                createInfo.subresourceRange.baseArrayLayer = 0;
                createInfo.subresourceRange.layerCount     = 1;

                if (vkCreateImageView(device, &createInfo, nullptr, &swapchainImageViews[i]) != VK_SUCCESS)
                {
                    throw std::runtime_error("Failed to create image views!");
                }
            }

            std::cout << "Image views created: " << swapchainImageViews.size() << std::endl;
        }

        void createRenderPass()
        {
            // Describe the color buffer attachment (one of the swapchain images)
            VkAttachmentDescription colorAttachment{};
            colorAttachment.format         = swapchainImageFormat;  // must match swapchain
            colorAttachment.samples        = VK_SAMPLE_COUNT_1_BIT; // no multisampling yet
            colorAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;   // clear on start
            colorAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;  // keep after render
            colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;        // don't care what it was
            colorAttachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;  // ready to present after

            // Subpass references the color attachment at index 0
            VkAttachmentReference colorAttachmentRef{};
            colorAttachmentRef.attachment = 0;
            colorAttachmentRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            // One subpass — this is where drawing happens
            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments    = &colorAttachmentRef;

            // Subpass dependency — makes sure the image is ready before we write to it
            VkSubpassDependency dependency{};
            dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
            dependency.dstSubpass    = 0;
            dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.srcAccessMask = 0;
            dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

            VkRenderPassCreateInfo renderPassInfo{};
            renderPassInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassInfo.attachmentCount = 1;
            renderPassInfo.pAttachments    = &colorAttachment;
            renderPassInfo.subpassCount    = 1;
            renderPassInfo.pSubpasses      = &subpass;
            renderPassInfo.dependencyCount = 1;
            renderPassInfo.pDependencies   = &dependency;

            if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create render pass!");
            }

            std::cout << "Render pass created successfully" << std::endl;
        }

        std::vector<char> readShaderFile(const std::string& filename)
        {
            std::ifstream file(filename, std::ios::ate | std::ios::binary);

            if (!file.is_open())
                throw std::runtime_error("Failed to open shader file: " + filename);

            size_t fileSize = (size_t)file.tellg();
            std::vector<char> buffer(fileSize);
            file.seekg(0);
            file.read(buffer.data(), fileSize);
            file.close();

            return buffer;
        }

        VkShaderModule createShaderModule(const std::vector<char>& code)
        {
            VkShaderModuleCreateInfo createInfo{};
            createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            createInfo.codeSize = code.size();
            createInfo.pCode    = reinterpret_cast<const uint32_t*>(code.data());

            VkShaderModule shaderModule;
            if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
                throw std::runtime_error("Failed to create shader module!");

            return shaderModule;
        }

        void createGraphicsPipeline()
        {
            // Load compiled shaders
            auto vertCode = readShaderFile("vert.spv");
            auto fragCode = readShaderFile("frag.spv");

            VkShaderModule vertModule = createShaderModule(vertCode);
            VkShaderModule fragModule = createShaderModule(fragCode);

            // Assign shaders to pipeline stages
            VkPipelineShaderStageCreateInfo vertStage{};
            vertStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            vertStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
            vertStage.module = vertModule;
            vertStage.pName  = "main";

            VkPipelineShaderStageCreateInfo fragStage{};
            fragStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            fragStage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            fragStage.module = fragModule;
            fragStage.pName  = "main";

            VkPipelineShaderStageCreateInfo shaderStages[] = { vertStage, fragStage };

            // Describe vertex input — one binding, one vec3 attribute
            VkVertexInputBindingDescription bindingDesc{};
            bindingDesc.binding   = 0;
            bindingDesc.stride    = sizeof(float) * 3;  // x, y, z
            bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            VkVertexInputAttributeDescription attrDesc{};
            attrDesc.binding  = 0;
            attrDesc.location = 0;
            attrDesc.format   = VK_FORMAT_R32G32B32_SFLOAT;  // vec3
            attrDesc.offset   = 0;

            VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
            vertexInputInfo.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertexInputInfo.vertexBindingDescriptionCount   = 1;
            vertexInputInfo.pVertexBindingDescriptions      = &bindingDesc;
            vertexInputInfo.vertexAttributeDescriptionCount = 1;
            vertexInputInfo.pVertexAttributeDescriptions    = &attrDesc;

            // Draw as points — each vertex is one body
            VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
            inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
            inputAssembly.primitiveRestartEnable = VK_FALSE;

            // Viewport covers the whole window
            VkViewport viewport{};
            viewport.x        = 0.0f;
            viewport.y        = 0.0f;
            viewport.width    = (float)swapchainExtent.width;
            viewport.height   = (float)swapchainExtent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;

            VkRect2D scissor{};
            scissor.offset = {0, 0};
            scissor.extent = swapchainExtent;

            VkPipelineViewportStateCreateInfo viewportState{};
            viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewportState.viewportCount = 1;
            viewportState.pViewports    = &viewport;
            viewportState.scissorCount  = 1;
            viewportState.pScissors     = &scissor;

            // Rasterizer
            VkPipelineRasterizationStateCreateInfo rasterizer{};
            rasterizer.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rasterizer.depthClampEnable        = VK_FALSE;
            rasterizer.rasterizerDiscardEnable = VK_FALSE;
            rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
            rasterizer.lineWidth               = 1.0f;
            rasterizer.cullMode                = VK_CULL_MODE_NONE;
            rasterizer.frontFace               = VK_FRONT_FACE_CLOCKWISE;
            rasterizer.depthBiasEnable         = VK_FALSE;

            // No multisampling
            VkPipelineMultisampleStateCreateInfo multisampling{};
            multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisampling.sampleShadingEnable  = VK_FALSE;
            multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            // Alpha blending — bodies add their brightness together
            VkPipelineColorBlendAttachmentState colorBlendAttachment{};
            colorBlendAttachment.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT |
                                                    VK_COLOR_COMPONENT_G_BIT |
                                                    VK_COLOR_COMPONENT_B_BIT |
                                                    VK_COLOR_COMPONENT_A_BIT;
            colorBlendAttachment.blendEnable         = VK_TRUE;
            colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;  // additive
            colorBlendAttachment.colorBlendOp        = VK_BLEND_OP_ADD;
            colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachment.alphaBlendOp        = VK_BLEND_OP_ADD;

            VkPipelineColorBlendStateCreateInfo colorBlending{};
            colorBlending.sType             = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            colorBlending.logicOpEnable     = VK_FALSE;
            colorBlending.attachmentCount   = 1;
            colorBlending.pAttachments      = &colorBlendAttachment;

            // Enable point size from vertex shader
            VkPipelineDynamicStateCreateInfo dynamicState{};
            std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
            dynamicState.pDynamicStates    = dynamicStates.data();

            VkPushConstantRange pushRange{};
            pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
            pushRange.offset     = 0;
            pushRange.size       = sizeof(float) * 4;

            // Pipeline layout — no uniforms yet
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
            pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount         = 0;
            pipelineLayoutInfo.pushConstantRangeCount = 1;           
            pipelineLayoutInfo.pPushConstantRanges    = &pushRange;  

            if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
                throw std::runtime_error("Failed to create pipeline layout!");

            // Finally create the pipeline
            VkGraphicsPipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineInfo.stageCount          = 2;
            pipelineInfo.pStages             = shaderStages;
            pipelineInfo.pVertexInputState   = &vertexInputInfo;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState      = &viewportState;
            pipelineInfo.pRasterizationState = &rasterizer;
            pipelineInfo.pMultisampleState   = &multisampling;
            pipelineInfo.pColorBlendState    = &colorBlending;
            pipelineInfo.pDynamicState       = &dynamicState;
            pipelineInfo.layout              = pipelineLayout;
            pipelineInfo.renderPass          = renderPass;
            pipelineInfo.subpass             = 0;

            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create graphics pipeline!");
            }

            std::cout << "Graphics pipeline created successfully" << std::endl;

            // Shader modules no longer needed after pipeline is created
            vkDestroyShaderModule(device, vertModule, nullptr);
            vkDestroyShaderModule(device, fragModule, nullptr);
        }

        void createFramebuffers()
        {
            swapchainFramebuffers.resize(swapchainImageViews.size());

            for (int i = 0; i < swapchainImageViews.size(); i++)
            {
                VkImageView attachments[] = { swapchainImageViews[i] };

                VkFramebufferCreateInfo framebufferInfo{};
                framebufferInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass      = renderPass;
                framebufferInfo.attachmentCount = 1;
                framebufferInfo.pAttachments    = attachments;
                framebufferInfo.width           = swapchainExtent.width;
                framebufferInfo.height          = swapchainExtent.height;
                framebufferInfo.layers          = 1;

                if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapchainFramebuffers[i]) != VK_SUCCESS)
                {
                    throw std::runtime_error("Failed to create framebuffer!");
                }
            }

            std::cout << "Framebuffers created: " << swapchainFramebuffers.size() << std::endl;
        }

        uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
        {
            VkPhysicalDeviceMemoryProperties memProps;
            vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

            for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
            {
                if ((typeFilter & (1 << i)) &&
                    (memProps.memoryTypes[i].propertyFlags & properties) == properties)
                    {
                        return i;
                    }
            }

            throw std::runtime_error("Failed to find suitable memory type!");
        }

        void createVertexBuffer()
        {
            VkDeviceSize bufferSize = sizeof(float) * 3 * NUM_BODIES;

            VkBufferCreateInfo bufferInfo{};
            bufferInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.size        = bufferSize;
            bufferInfo.usage       = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            if (vkCreateBuffer(device, &bufferInfo, nullptr, &vertexBuffer) != VK_SUCCESS)
                throw std::runtime_error("Failed to create vertex buffer!");

            VkMemoryRequirements memReqs;
            vkGetBufferMemoryRequirements(device, vertexBuffer, &memReqs);

            VkMemoryAllocateInfo allocInfo{};
            allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize  = memReqs.size;
            allocInfo.memoryTypeIndex = findMemoryType(
                memReqs.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            );

            if (vkAllocateMemory(device, &allocInfo, nullptr, &vertexBufferMemory) != VK_SUCCESS)
                throw std::runtime_error("Failed to allocate vertex buffer memory!");

            vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0);

            std::cout << "Vertex buffer created" << std::endl;
        }

        void uploadPositions(const std::vector<float>& x,
                     const std::vector<float>& y,
                     const std::vector<float>& z)
        {
            VkDeviceSize bufferSize = sizeof(float) * 3 * NUM_BODIES;

            // Interleave x,y,z into one buffer
            std::vector<float> interleaved(NUM_BODIES * 3);
            for (int i = 0; i < NUM_BODIES; i++)
            {
                interleaved[i * 3 + 0] = x[i];
                interleaved[i * 3 + 1] = y[i];
                interleaved[i * 3 + 2] = z[i];
            }

            void* data;
            vkMapMemory(device, vertexBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, interleaved.data(), bufferSize);
            vkUnmapMemory(device, vertexBufferMemory);
        }

        void createCommandPool()
        {
            QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

            VkCommandPoolCreateInfo poolInfo{};
            poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.queueFamilyIndex = indices.graphicsFamily;
            poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

            if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
                throw std::runtime_error("Failed to create command pool!");

            // One command buffer per swapchain image
            commandBuffers.resize(swapchainFramebuffers.size());

            VkCommandBufferAllocateInfo allocInfo{};
            allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool        = commandPool;
            allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

            if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
                throw std::runtime_error("Failed to allocate command buffers!");

            std::cout << "Command pool and buffers created" << std::endl;
        }

        void createSyncObjects()
        {
            VkSemaphoreCreateInfo semaphoreInfo{};
            semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

            VkFenceCreateInfo fenceInfo{};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // start signaled so first frame doesn't hang

            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphore) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphore) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFence) != VK_SUCCESS)
                throw std::runtime_error("Failed to create sync objects!");

            std::cout << "Sync objects created" << std::endl;
        }

        void drawFrame()
        {
            // Wait for previous frame to finish
            vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
            vkResetFences(device, 1, &inFlightFence);

            // Get next swapchain image
            uint32_t imageIndex;
            vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
                                imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

            // Record command buffer
            VkCommandBuffer cmd = commandBuffers[imageIndex];
            vkResetCommandBuffer(cmd, 0);

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            vkBeginCommandBuffer(cmd, &beginInfo);

            VkBufferMemoryBarrier barrier{};
            barrier.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
            barrier.buffer        = vertexBuffer;
            barrier.size          = VK_WHOLE_SIZE;

            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                0, 0, nullptr, 1, &barrier, 0, nullptr);

            VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};  // black background

            VkRenderPassBeginInfo renderPassInfo{};
            renderPassInfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass        = renderPass;
            renderPassInfo.framebuffer       = swapchainFramebuffers[imageIndex];
            renderPassInfo.renderArea.offset = {0, 0};
            renderPassInfo.renderArea.extent = swapchainExtent;
            renderPassInfo.clearValueCount   = 1;
            renderPassInfo.pClearValues      = &clearColor;

            vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

            float pushData[4] = { camZoom, camX, camY, 0.0f };
            vkCmdPushConstants(cmd, pipelineLayout,
                   VK_SHADER_STAGE_VERTEX_BIT,
                   0, sizeof(pushData), pushData);

            // Set dynamic viewport and scissor
            VkViewport viewport{};
            viewport.x        = 0.0f;
            viewport.y        = 0.0f;
            viewport.width    = (float)swapchainExtent.width;
            viewport.height   = (float)swapchainExtent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(cmd, 0, 1, &viewport);

            VkRect2D scissor{};
            scissor.offset = {0, 0};
            scissor.extent = swapchainExtent;
            vkCmdSetScissor(cmd, 0, 1, &scissor);

            VkBuffer vertexBuffers[] = { vertexBuffer };
            VkDeviceSize offsets[]   = { 0 };
            vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
            vkCmdDraw(cmd, NUM_BODIES, 1, 0, 0);    // draw all bodies

            vkCmdEndRenderPass(cmd);
            vkEndCommandBuffer(cmd);

            // Submit
            VkSemaphore waitSemaphores[]   = { imageAvailableSemaphore };
            VkSemaphore signalSemaphores[] = { renderFinishedSemaphore };
            VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

            VkSubmitInfo submitInfo{};
            submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.waitSemaphoreCount   = 1;
            submitInfo.pWaitSemaphores      = waitSemaphores;
            submitInfo.pWaitDstStageMask    = waitStages;
            submitInfo.commandBufferCount   = 1;
            submitInfo.pCommandBuffers      = &cmd;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores    = signalSemaphores;

            vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFence);

            // Present
            VkPresentInfoKHR presentInfo{};
            presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores    = signalSemaphores;
            presentInfo.swapchainCount     = 1;
            presentInfo.pSwapchains        = &swapchain;
            presentInfo.pImageIndices      = &imageIndex;

            vkQueuePresentKHR(presentQueue, &presentInfo);
        }

        void mainLoop(std::function<void()> simulateStep)   // add this
        {
            while (!glfwWindowShouldClose(window))
            {
                glfwPollEvents();
                simulateStep();       // run one OpenCL step
                drawFrame();  // render result
            }

            vkDeviceWaitIdle(device);
        }

        void cleanup() 
        {
            vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
            vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
            vkDestroyFence(device, inFlightFence, nullptr);

            vkDestroyCommandPool(device, commandPool, nullptr);

            for (auto fb : swapchainFramebuffers)
            {
                vkDestroyFramebuffer(device, fb, nullptr);
            }
                
            vkDestroyPipeline(device, graphicsPipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

            vkDestroyRenderPass(device, renderPass, nullptr); 

            for (auto imageView : swapchainImageViews) 
            {
                vkDestroyImageView(device, imageView, nullptr);
            }      

            vkDestroySwapchainKHR(device, swapchain, nullptr);

            vkDestroyDevice(device, nullptr);

            vkDestroySurfaceKHR(instance, surface, nullptr); 
            vkDestroyInstance(instance, nullptr);

            glfwDestroyWindow(window);

            glfwTerminate();
        }
};


#pragma region FILE_READ_HELPER

std::string loadFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Cannot open file: " << path << std::endl;
        exit(1);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

#pragma endregion

int main()
{
    srand(time(NULL));

#pragma region PLATFORM_AND_DEVICE

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    // Get GPU device
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];

    std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

#pragma endregion

#pragma region CONTEXT

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

#pragma endregion

#pragma region KERNEL__AND_HEADER_FOR_KERNEL

    std::string configSrc = loadFile("variables.hpp");
    std::string kernelSrc = loadFile("kernels.cl");

    cl::Program::Sources sources;
    sources.push_back(configSrc);
    sources.push_back(kernelSrc);

    cl::Program program(context, sources);

    try
    {
        program.build({ device });
    }
    catch (const cl::Error&)
    {
        std::cerr << "Build error:\n"
                    << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
                    << std::endl;
        return 1;
    }

    cl::Kernel resetBBoxKernel(program, "resetBBoxKernel");
    cl::Kernel boundingBoxKernel(program, "boundingBoxKernel");
    cl::Kernel initTreeKernel(program, "initTreeKernel");
    cl::Kernel insertKernel(program, "insertBodiesKernel");
    cl::Kernel comKernel(program, "computeCOMKernel");
    cl::Kernel forceKernel(program, "forceKernel");
    cl::Kernel integKernel(program, "integrationKernel");
    cl::Kernel writePositionsKernel(program, "writePositionsInterleaved");

#pragma endregion

#pragma region GPU_BUFFERS

#pragma region BODY_BUFFERS

    cl::Buffer buf_x(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_y(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_z(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_vx(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_vy(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_vz(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_fx(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_fy(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_fz(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_mass(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));

#pragma endregion

#pragma region OCTREE

    cl::Buffer buf_child    (context, CL_MEM_READ_WRITE, MAX_NODE*8*sizeof(int)  );
    cl::Buffer buf_nodeX    (context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(float)  );
    cl::Buffer buf_nodeY    (context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(float)  );
    cl::Buffer buf_nodeZ    (context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(float)  );
    cl::Buffer buf_nodeMass (context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(float)  );
    cl::Buffer buf_nodeCount(context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(int)    );
    cl::Buffer buf_nodeSize (context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(float)  );
    cl::Buffer buf_nextNode (context, CL_MEM_READ_WRITE, sizeof(int)             );

#pragma endregion

    cl::Buffer buf_bbox(context, CL_MEM_READ_WRITE, 6*sizeof(float));
    cl::Buffer buf_flag(context, CL_MEM_READ_WRITE, sizeof(int)    );

#pragma endregion

#pragma region INIT

    // Initialise bodies on CPU
    std::vector<float> h_x(NUM_BODIES), h_y(NUM_BODIES), h_z(NUM_BODIES);
    std::vector<float> h_vx(NUM_BODIES), h_vy(NUM_BODIES), h_vz(NUM_BODIES);
    std::vector<float> h_fx(NUM_BODIES), h_fy(NUM_BODIES), h_fz(NUM_BODIES);
    std::vector<float> h_mass(NUM_BODIES);

    for (int i = 0; i < NUM_BODIES; i++)
    {
        // Random angle and radius in a disk
        float angle  = ((float)rand() / RAND_MAX) * 2.0f * 3.14159f;
        float radius = ((float)rand() / RAND_MAX) * (SPAWN_RANGE / 2.0f);

        // Thin disk — z is small compared to x,y
        float diskHeight = SPAWN_RANGE;

        h_x[i] = radius * cos(angle);
        h_y[i] = radius * sin(angle);
        h_z[i] = (float)((rand() % (int)diskHeight) - diskHeight / 2.0f);

        // Orbital velocity — bodies orbit the center
        float totalMass = (float)NUM_BODIES * 12000.0f;

        // Never let radius go below this for velocity calc
        float safeRadius = fmax(radius, SPAWN_RANGE / 10.0f);
        float orbitSpeed = sqrt(G * totalMass / safeRadius) * 0.3f;

        h_vx[i] =  sin(angle) * orbitSpeed;
        h_vy[i] = -cos(angle) * orbitSpeed;
        h_vz[i] = 0.0f;

        h_mass[i] = 10000.0f;
    }

    // Upload to GPU
    queue.enqueueWriteBuffer(buf_x,    CL_TRUE, 0, NUM_BODIES*sizeof(float), h_x.data()   );
    queue.enqueueWriteBuffer(buf_y,    CL_TRUE, 0, NUM_BODIES*sizeof(float), h_y.data()   );
    queue.enqueueWriteBuffer(buf_z,    CL_TRUE, 0, NUM_BODIES*sizeof(float), h_z.data()   );
    queue.enqueueWriteBuffer(buf_vx,   CL_TRUE, 0, NUM_BODIES*sizeof(float), h_vx.data()  );
    queue.enqueueWriteBuffer(buf_vy,   CL_TRUE, 0, NUM_BODIES*sizeof(float), h_vy.data()  );
    queue.enqueueWriteBuffer(buf_vz,   CL_TRUE, 0, NUM_BODIES*sizeof(float), h_vz.data()  );
    queue.enqueueWriteBuffer(buf_fx,   CL_TRUE, 0, NUM_BODIES*sizeof(float), h_fx.data()  );
    queue.enqueueWriteBuffer(buf_fy,   CL_TRUE, 0, NUM_BODIES*sizeof(float), h_fy.data()  );
    queue.enqueueWriteBuffer(buf_fz,   CL_TRUE, 0, NUM_BODIES*sizeof(float), h_fz.data()  );
    queue.enqueueWriteBuffer(buf_mass, CL_TRUE, 0, NUM_BODIES*sizeof(float), h_mass.data());

#pragma endregion

#pragma region LOOP

    cl::NDRange global(NUM_BODIES);
    cl::NDRange local(THREADS);
    cl::NDRange globalTree(MAX_NODE);
    cl::NDRange one(1);

    Simulation sim;

    try
    {
        sim.init();

        // Get pointer to Vulkan's vertex buffer memory
        void* mappedPtr = sim.getMappedVertexBuffer();

        // Create OpenCL buffer pointing to Vulkan memory — must exist before lambda
        cl::Buffer buf_pos(
            context,
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
            sizeof(float) * 3 * NUM_BODIES,
            mappedPtr
        );
        std::cout << "Shared memory buffer created" << std::endl;

        // Lambda defined AFTER buf_pos — captures the correct one
        auto simulateStep = [&]()
        {
            // 1. Reset bbox
            resetBBoxKernel.setArg(0, buf_bbox);
            queue.enqueueNDRangeKernel(resetBBoxKernel, cl::NullRange, one, one);
            queue.finish();

            // 2. Bounding box
            boundingBoxKernel.setArg(0, buf_bbox);
            boundingBoxKernel.setArg(1, buf_x);
            boundingBoxKernel.setArg(2, buf_y);
            boundingBoxKernel.setArg(3, buf_z);
            queue.enqueueNDRangeKernel(boundingBoxKernel, cl::NullRange, global, local);
            queue.finish();

            // 3. Init tree
            initTreeKernel.setArg(0, buf_child);
            initTreeKernel.setArg(1, buf_nodeX);
            initTreeKernel.setArg(2, buf_nodeY);
            initTreeKernel.setArg(3, buf_nodeZ);
            initTreeKernel.setArg(4, buf_nodeMass);
            initTreeKernel.setArg(5, buf_nodeCount);
            initTreeKernel.setArg(6, buf_nodeSize);
            initTreeKernel.setArg(7, buf_nextNode);
            initTreeKernel.setArg(8, buf_bbox);
            queue.enqueueNDRangeKernel(initTreeKernel, cl::NullRange, globalTree, local);
            queue.finish();

            // 4. Insert bodies
            insertKernel.setArg(0, buf_child);
            insertKernel.setArg(1, buf_nodeX);
            insertKernel.setArg(2, buf_nodeY);
            insertKernel.setArg(3, buf_nodeZ);
            insertKernel.setArg(4, buf_nodeMass);
            insertKernel.setArg(5, buf_nodeCount);
            insertKernel.setArg(6, buf_nodeSize);
            insertKernel.setArg(7, buf_nextNode);
            insertKernel.setArg(8, buf_x);
            insertKernel.setArg(9, buf_y);
            insertKernel.setArg(10, buf_z);
            insertKernel.setArg(11, buf_mass);
            queue.enqueueNDRangeKernel(insertKernel, cl::NullRange, global, local);
            queue.finish();

            // 5. COM
            comKernel.setArg(0, buf_child);
            comKernel.setArg(1, buf_nodeX);
            comKernel.setArg(2, buf_nodeY);
            comKernel.setArg(3, buf_nodeZ);
            comKernel.setArg(4, buf_nodeMass);
            comKernel.setArg(5, buf_nodeCount);
            comKernel.setArg(6, buf_x);
            comKernel.setArg(7, buf_y);
            comKernel.setArg(8, buf_z);
            comKernel.setArg(9, buf_mass);
            comKernel.setArg(10, buf_nextNode);
            queue.enqueueNDRangeKernel(comKernel, cl::NullRange, cl::NDRange(1), cl::NDRange(1));
            queue.finish();

            // 6. Forces
            forceKernel.setArg(0, buf_child);
            forceKernel.setArg(1, buf_nodeX);
            forceKernel.setArg(2, buf_nodeY);
            forceKernel.setArg(3, buf_nodeZ);
            forceKernel.setArg(4, buf_nodeMass);
            forceKernel.setArg(5, buf_nodeSize);
            forceKernel.setArg(6, buf_nextNode);
            forceKernel.setArg(7, buf_x);
            forceKernel.setArg(8, buf_y);
            forceKernel.setArg(9, buf_z);
            forceKernel.setArg(10, buf_fx);
            forceKernel.setArg(11, buf_fy);
            forceKernel.setArg(12, buf_fz);
            forceKernel.setArg(13, buf_mass);
            queue.enqueueNDRangeKernel(forceKernel, cl::NullRange, global, local);
            queue.finish();

            // 7. Integration
            integKernel.setArg(0, buf_x);
            integKernel.setArg(1, buf_y);
            integKernel.setArg(2, buf_z);
            integKernel.setArg(3, buf_vx);
            integKernel.setArg(4, buf_vy);
            integKernel.setArg(5, buf_vz);
            integKernel.setArg(6, buf_fx);
            integKernel.setArg(7, buf_fy);
            integKernel.setArg(8, buf_fz);
            integKernel.setArg(9, buf_mass);
            queue.enqueueNDRangeKernel(integKernel, cl::NullRange, global, local);
            queue.finish();

            // 8. Write positions directly into Vulkan vertex buffer memory
            writePositionsKernel.setArg(0, buf_x);
            writePositionsKernel.setArg(1, buf_y);
            writePositionsKernel.setArg(2, buf_z);
            writePositionsKernel.setArg(3, buf_pos);
            queue.enqueueNDRangeKernel(writePositionsKernel, cl::NullRange, global, local);
            queue.finish();
        };

        sim.loop(simulateStep);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    #pragma endregion

    return 0;
}