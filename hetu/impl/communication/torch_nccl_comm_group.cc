// #include "hetu/impl/communication/torch_nccl_comm_group.h"
// #include "hetu/impl/communication/mpi_comm_group.h" // For GetWorldRank, GetWorldSize, GetGroupRank
// #include "hetu/impl/stream/CUDAStream.h"
// #include "hetu/impl/utils/ndarray_utils.h"
// #include "hetu/graph/graph.h"
// #include "hetu/graph/executable_graph.h"
// #include "hetu/utils/task_queue.h"
// #include "hetu/core/ndarray_storage.h"

// #include <torch/torch.h>
// #include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
// #include <torch/csrc/distributed/c10d/FileStore.hpp>
// #include <torch/csrc/distributed/c10d/TCPStore.hpp>
// #include <torch/csrc/distributed/c10d/Types.hpp>
// #include <c10/cuda/CUDAStream.h>

// #include <numeric>
// #include <mutex>
// #include <vector>
// #include <map>
// #include <memory>
// #include <stdexcept> // For runtime_error
// #include <cstdlib> // For std::getenv
// #include <algorithm> // For std::sort
// #include <chrono> // For timeouts

// namespace hetu {
// namespace impl {
// namespace comm {

// using hetu::operator<<;

// namespace {


// static std::once_flag torch_nccl_init_flag;
// static std::mutex torch_nccl_create_group_mutex;
// static std::vector<
//   std::vector<std::map<std::vector<int>, NCCLCommunicationGroup>>>
//   torch_nccl_comm_groups(
//     (HT_NUM_STREAMS_PER_DEVICE) + 1,
//     std::vector<std::map<std::vector<int>, NCCLCommunicationGroup>>(HT_MAX_GPUS_RUN_TIME));
// static std::vector<std::vector<NCCLCommunicationGroup>>
//   worldwide_torch_nccl_comm_groups(
//     (HT_NUM_STREAMS_PER_DEVICE) + 1,
//     std::vector<NCCLCommunicationGroup>(HT_MAX_GPUS_RUN_TIME));
// static int id = 0;

// int get_id() {
//   id++;
//   return id;
// }


// static void TorchNCCL_Init_Once() {
//   std::call_once(torch_nccl_init_flag, []() {
//     // register exit handler
//     HT_ASSERT(std::atexit([]() {
//                 std::lock_guard<std::mutex> lock(torch_nccl_create_group_mutex);
//                 HT_LOG_DEBUG << "Destructing TorchNCCL comm groups...";
//                 torch_nccl_comm_groups.clear();
//                 worldwide_torch_nccl_comm_groups.clear();
//                 HT_LOG_DEBUG << "Destructed TorchNCCL comm groups";
//               }) == 0)
//       << "Failed to register the exit function for NCCL.";
//   });
// }



// } // namespace anonymous

// // ================== NCCLCommunicationGroupDef ==================

// NCCLCommunicationGroupDef::NCCLCommunicationGroupDef(
//     const std::vector<int>& world_ranks, const Stream& stream)
// : CommunicationGroupDef(world_ranks, stream) {

//     TorchNCCL_Init_Once(); // 确保退出处理程序已注册

//     HT_ASSERT(_stream.device().is_cuda())
//         << "NCCL communication group must be initialized with "
//         << "a stream related with CUDA. Got " << _stream << ".";
//     int world_size = GetWorldSize();
//     HT_ASSERT(_world_ranks.back() < world_size)
//         << "Invalid ranks " << _world_ranks << " for world size " << world_size
//         << ".";

//     if (_world_ranks.size() == static_cast<size_t>(world_size)) {
//         _rank = GetWorldRank();
//         _size = world_size;
//     } else {
//         _rank = GetGroupRank(_world_ranks);
//         _size = _world_ranks.size();
//         HT_ASSERT(_rank != -1) << "The current rank " << GetWorldRank()
//                             << " is not included in the group " << _world_ranks
//                             << ".";
//     }
//     HT_ASSERT(_rank >= 0 && _rank < _size)
//         << "Failed to get rank and/or size. "
//         << "(Got rank " << _rank << " and size " << _size << ".)";


//     // --- 初始化 PyTorch Process Group ---

//     // 1. 基于组 ranks 创建唯一的存储标识符
//     std::string group_id_str;
//     for (size_t i = 0; i < _world_ranks.size(); ++i) {
//         group_id_str += std::to_string(_world_ranks[i]);
//         if (i < _world_ranks.size() - 1) {
//             group_id_str += "_";
//         }
//     }
//     // 如果 group_id_str 太长，可能需要进行哈希处理
//     // TODO: 考虑更健壮的临时文件路径生成和管理
//     std::string store_path = "/home/gehao/njw1123/hetu_add_torch/hetu_torch_pg_store_" + group_id_str + "_" + std::to_string(get_id());

//     // 2. 创建 FileStore
//     try {
//          // 使用组大小 (_size) 而不是 world_size
//         _store = c10::make_intrusive<c10d::FileStore>(store_path, _size);
//     } catch (const std::exception& e) {
//          // 提供更详细的错误信息
//          HT_RUNTIME_ERROR << "Failed to create FileStore at path '" << store_path
//                       << "' for group size " << _size << " (world ranks: " << _world_ranks
//                       << ", current world rank: " << _rank << ", group rank: " << _rank
//                       << "). Check filesystem permissions and path validity. Error: " << e.what();
//          throw; // 记录日志后重新抛出异常
//     }


//     // 3. 创建 ProcessGroupNCCL 选项
//     // auto opts = c10::make_intrusive<c10d::ProcessGroupNCCL::Options>(c10d::ProcessGroupNCCL::Options::kDefaultTimeout);
//     auto opts = c10d::ProcessGroupNCCL::Options::create();

//     // 4. 创建 ProcessGroupNCCL 实例
//     try {
//         // 确保操作在正确的设备上下文中进行
//         hetu::cuda::CUDADeviceGuard guard(_stream.device_index());
//         auto torch_stream = GetTorchCudaStream(_stream);
//         c10::cuda::CUDAStreamGuard stream_guard(torch_stream);
//         // 使用组内的 rank 和 size
//         std::cout << "create process group" << std::endl;
//         _process_group = c10::make_intrusive<c10d::ProcessGroupNCCL>(_store, _rank, _size, opts);
//     } catch (const std::exception& e) {
//          HT_RUNTIME_ERROR << "Failed to create ProcessGroupNCCL for group rank " << _rank
//                       << " size " << _size << " (world ranks: " << _world_ranks
//                       << ", current world rank: " << _rank
//                       << "). Check NCCL setup and CUDA environment. Error: " << e.what();
//          throw; // 记录日志后重新抛出异常
//     }

//     HT_LOG_DEBUG << "Successfully initialized TorchNCCL comm group (PG rank " << _rank << "/" << _size
//                  << ") for world ranks: " << _world_ranks << " on device " << _stream.device_index()
//                  << " using store: " << store_path;
// }


// NCCLCommunicationGroupDef::~NCCLCommunicationGroupDef() {
//     Sync();
//     HT_LOG_DEBUG << "Destroying TorchNCCL comm group wrapper (PG rank " << _rank << "/" << _size << ").";
// }

// void NCCLCommunicationGroupDef::Sync() {
//   CUDAStream(_stream).Sync();
// }

// void NCCLCommunicationGroupDef::Barrier(bool sync) {


//     if (!_process_group) {
//         HT_RUNTIME_ERROR << "在调用barrier时_process_group为null";
//         return;
//     }

//     c10d::BarrierOptions opts;

//     // Ensure operations are on the correct stream context
//     CUDAStream cuda_stream(_stream);
//     hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
//     std::cout << "start get torch stream" << std::endl;
//     std::cout << "stream id: " << _stream << std::endl;
//     try {
//         std::cout << "start get torch stream" << std::endl; 
//         auto torch_stream = GetTorchCudaStream(_stream);
//         std::cout << "start stream guard" << std::endl;
//         c10::cuda::CUDAStreamGuard stream_guard(torch_stream);
//         std::cout << "stream guard done" << std::endl;
        
//         // 调用barrier并捕获可能的异常
//         std::cout << "_process_group: " << _process_group << std::endl;
//         auto work = _process_group->barrier(opts);
//         std::cout << "barrier done" << std::endl;
//         if (sync) {
//             work->wait();
//         }
//     } catch (const std::exception& e) {
//         HT_RUNTIME_ERROR << "Barrier操作失败: " << e.what();
//     }
// }


// void NCCLCommunicationGroupDef::Broadcast(NDArray& data, int broadcaster_world_rank) {
//     HT_ASSERT_CUDA_DEVICE(data);

//     // 设置CUDA和Stream守卫，确保操作在正确的设备上执行
//     auto& local_device = hetu::impl::comm::GetLocalDevice();
//     CUDAStream cuda_stream(_stream);
//     hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
//     c10::Device torch_device(c10::DeviceType::CUDA, local_device.index());
//     c10::cuda::CUDAStreamGuard stream_guard(GetTorchCudaStream(_stream));   
//     c10::cuda::CUDAGuard device_guard(torch_device);


//     int root_rank = world_to_group_rank(broadcaster_world_rank);

//     auto tensor = TransNDArray2Tensor(data);
//     std::vector<torch::Tensor> tensors = {tensor}; // Ops take a vector

//     c10d::BroadcastOptions opts;
//     opts.rootRank = root_rank;
//     opts.rootTensor = 0; 

//     auto work = _process_group->broadcast(tensors, opts);

//     NDArray::MarkUsedBy(data, _stream); // Mark NDArray used on this stream
// }

// void NCCLCommunicationGroupDef::AllReduce(const NDArray& input, NDArray& output, ReductionType red_type) {
//     HT_ASSERT_CUDA_DEVICE(input);
//     HT_ASSERT_CUDA_DEVICE(output);
//     HT_ASSERT_SAME_SHAPE(input, output);
//     HT_ASSERT_SAME_DTYPE(input, output);

//     // 设置CUDA和Stream守卫，确保操作在正确的设备上执行
//     auto& local_device = hetu::impl::comm::GetLocalDevice();
//     CUDAStream cuda_stream(_stream);
//     hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
//     c10::Device torch_device(c10::DeviceType::CUDA, local_device.index());
//     c10::cuda::CUDAStreamGuard stream_guard(GetTorchCudaStream(_stream));   
//     c10::cuda::CUDAGuard device_guard(torch_device);

//     // 将输入数据复制到输出张量（如果它们不是同一个内存）
//     if (input->raw_data_ptr() != output->raw_data_ptr()) {
//         NDArray::copy(input, _stream.stream_index(), output);
//     }

//     // 只创建一个包含输出张量的向量
//     auto output_tensor = TransNDArray2Tensor(output);
//     std::vector<torch::Tensor> tensors = {output_tensor};

//     c10d::AllreduceOptions opts;
//     opts.reduceOp = ToTorchRedType(red_type);

//     // 调用 PyTorch allreduce（现在只传一个张量）
//     auto work = _process_group->allreduce(tensors, opts);
    

//     NDArray::MarkUsedBy({input, output}, _stream);
// }


// void NCCLCommunicationGroupDef::AllReduceCoalesce(
//     const NDArrayList& inputs, NDArrayList& outputs,
//     NDArray /*contiguous_buffers*/, // Torch PG handles coalescing internally
//     ReductionType red_type)
// {
//     HT_RUNTIME_ERROR << "AllReduceCoalesce is not implemented for TorchNCCL.";
// }

// void NCCLCommunicationGroupDef::AlltoAll(const NDArray& input, NDArray& output) {
//     HT_NOT_IMPLEMENTED << "AlltoAll is not implemented for TorchNCCL.";
// }

// void NCCLCommunicationGroupDef::Reduce(const NDArray& input, NDArray& output, int reducer_world_rank, ReductionType red_type) {
//     HT_NOT_IMPLEMENTED << "Reduce is not implemented for TorchNCCL.";
// }
// void NCCLCommunicationGroupDef::AllGather(const NDArray& input,
//                                           NDArray& output, int32_t gather_dim) {
//     HT_ASSERT_CUDA_DEVICE(input);
//     HT_ASSERT_CUDA_DEVICE(output);
//     HT_ASSERT_SAME_DTYPE(input, output);
//     HT_ASSERT(input->is_contiguous() && output->is_contiguous())
//         << "Input and output tensors must be contiguous.";

//     size_t input_size = input->numel();
//     size_t output_size = output->numel();
//     HT_ASSERT(input->shape(gather_dim) * _size == output->shape(gather_dim) &&
//               input_size * _size == output_size)
//       << "Invalid shapes for AllGather: "
//       << "(send) " << input->shape() << " vs. "
//       << "(recv) " << output->shape() << ".";
//     input_size = (input->dtype() == kNFloat4 || input->dtype() == kFloat4)  
//               ? (input_size + 1) / 2
//               : input_size;
//     output_size = (output->dtype() == kNFloat4 || output->dtype() == kFloat4)  
//                 ? (output_size + 1) / 2
//                 : output_size;
//     if (output_size == 0) {
//       return;
//     }

//     // 设置CUDA和Stream守卫
//     CUDAStream cuda_stream(_stream);
//     hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
//     c10::Device torch_device(c10::DeviceType::CUDA, _stream.device().index());
//     c10::cuda::CUDAStreamGuard stream_guard(GetTorchCudaStream(_stream));
//     c10::cuda::CUDAGuard device_guard(torch_device);

//     // 转换 NDArray 到 Tensor
//     std::cout << "input: " << input << std::endl;
//     std::cout << "output: " << output << std::endl;
//     auto input_tensor = TransNDArray2Tensor(input);
//     auto output_tensor = TransNDArray2Tensor(output);

//     c10d::AllgatherOptions opts;
//     auto work = _process_group->_allgather_base(output_tensor, input_tensor, opts);

//     // 标记输出 NDArray 已被此流使用
//     NDArray::MarkUsedBy({input, output}, _stream);

// }

// void NCCLCommunicationGroupDef::ReduceScatter(const NDArray& input,
//                                                   NDArray& output, int32_t scatter_dim,
//                                                   ReductionType red_type) {
//     HT_ASSERT_CUDA_DEVICE(input);
//     HT_ASSERT_CUDA_DEVICE(output);
//     HT_ASSERT_SAME_DTYPE(input, output);
//     HT_ASSERT(input->is_contiguous() && output->is_contiguous())
//         << "Input and output tensors must be contiguous.";

//     size_t input_size = input->numel();
//     size_t output_size = output->numel();

//     // 输入张量的元素总数必须是输出张量元素总数的 _size 倍
//     HT_ASSERT(input_size == output_size * _size)
//       << "Invalid shapes for ReduceScatter: input size (" << input_size
//       << ") must be group size (" << _size << ") times output size (" << output_size << "). "
//       << "(send) " << input->shape() << " vs. "
//       << "(recv) " << output->shape() << ".";

//     // Note: 不再需要像原始 NCCL 那样为 FP4/BF16 调整大小，假设 PyTorch Tensor 会处理

//     if (output_size == 0) {
//       return;
//     }

//     // 设置CUDA和Stream守卫
//     CUDAStream cuda_stream(_stream);
//     hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
//     c10::Device torch_device(c10::DeviceType::CUDA, _stream.device().index());
//     c10::cuda::CUDAStreamGuard stream_guard(GetTorchCudaStream(_stream));
//     c10::cuda::CUDAGuard device_guard(torch_device);

//     // 转换 NDArray 到 Tensor
//     auto input_tensor = TransNDArray2Tensor(input);
//     auto output_tensor = TransNDArray2Tensor(output);

//     // 设置 ReduceScatter 选项
//     c10d::ReduceScatterOptions opts;
//     opts.reduceOp = ToTorchRedType(red_type);

//     // 调用 PyTorch 的 _reduce_scatter_base
//     auto work = _process_group->_reduce_scatter_base(output_tensor, input_tensor, opts);

//     NDArray::MarkUsedBy({input, output}, _stream);
// }

// void NCCLCommunicationGroupDef::Gather(const NDArray& input, NDArray& output,
//                                            int gatherer) {
//     HT_NOT_IMPLEMENTED << "Gather is not implemented for TorchNCCL.";
// }

// void NCCLCommunicationGroupDef::Scatter(const NDArray& input, NDArray& output,
//                                             int scatterer) {
//     HT_NOT_IMPLEMENTED << "Scatter is not implemented for TorchNCCL.";
// }

// void NCCLCommunicationGroupDef::Send(const NDArray& data, int receiver) {
//     HT_NOT_IMPLEMENTED << "Send is not implemented for TorchNCCL.";
// }

// void NCCLCommunicationGroupDef::Recv(NDArray& data, int sender) {
//     HT_NOT_IMPLEMENTED << "Recv is not implemented for TorchNCCL.";
// }

// CommTask NCCLCommunicationGroupDef::ISend(const NDArray& data, int receiver) {
//     HT_NOT_IMPLEMENTED << "ISend is not implemented for TorchNCCL.";
//     // 返回一个空的 CommTask 以满足返回类型要求
//     return CommTask();
// }

// CommTask NCCLCommunicationGroupDef::IRecv(NDArray& data, int sender) {
//     HT_NOT_IMPLEMENTED << "IRecv is not implemented for TorchNCCL.";
//     // 返回一个空的 CommTask 以满足返回类型要求
//     return CommTask();
// }

// void NCCLCommunicationGroupDef::BatchedISendIRecv(
//     const std::vector<CommTask>& tasks) {
//     HT_NOT_IMPLEMENTED << "BatchedISendIRecv is not implemented for TorchNCCL.";
// }




// // ================== NCCLCommunicationGroup ==================



// NCCLCommunicationGroup&
// NCCLCommunicationGroup::GetOrCreate(const std::vector<int>& world_ranks,
//                                     const Stream& stream) {
//   HT_ASSERT(stream.device().is_cuda())
//     << "The argument \"stream\" for "
//     << "NCCLCommunicationGroup::GetOrCreate "
//     << "must be a CUDA stream. Got " << stream << ".";
//   // Note: stream id could be -1, we shall shift it by one when accessing
//   int stream_id = static_cast<int>(stream.stream_index());
//   int device_id = static_cast<int>(stream.device_index());

//   TorchNCCL_Init_Once();

//   HT_ASSERT(world_ranks.empty() ||
//             CommunicationGroupDef::IsRanksValid(world_ranks))
//     << "Invalid world ranks: " << world_ranks;
//   auto world_size = static_cast<size_t>(GetWorldSize());

//   if (world_ranks.empty() ||
//       static_cast<int>(world_ranks.size()) == world_size) {
//     if (!worldwide_torch_nccl_comm_groups[stream_id + 1][device_id].is_defined()) {
//       std::unique_lock<std::mutex> lock(torch_nccl_create_group_mutex);
//       // double check for thread-safety
//       if (!worldwide_torch_nccl_comm_groups[stream_id + 1][device_id].is_defined()) {
//         std::vector<int> all_world_ranks(world_size);
//         std::iota(all_world_ranks.begin(), all_world_ranks.end(), 0);
//         worldwide_torch_nccl_comm_groups[stream_id + 1][device_id] =
//           NCCLCommunicationGroup(all_world_ranks, stream);
//       }
//     }
//     return worldwide_torch_nccl_comm_groups[stream_id + 1][device_id];
//   } else {
//     HT_ASSERT(GetGroupRank(world_ranks) != -1)
//       << "Cannot get comm group " << world_ranks << " on rank "
//       << GetWorldRank() << ".";
//     auto it = torch_nccl_comm_groups[stream_id + 1][device_id].find(world_ranks);
//     if (it == torch_nccl_comm_groups[stream_id + 1][device_id].end()) {
//       std::unique_lock<std::mutex> lock(torch_nccl_create_group_mutex);
//       // double check for thread-safety
//       it = torch_nccl_comm_groups[stream_id + 1][device_id].find(world_ranks);
//       if (it == torch_nccl_comm_groups[stream_id + 1][device_id].end()) {
//         HT_LOG_INFO << "Create NCCLCommunicationGroup for world ranks " << world_ranks << " on stream " << stream << " begin...";
//         NCCLCommunicationGroup comm_group(world_ranks, stream);
//         HT_LOG_INFO << "Create NCCLCommunicationGroup for world ranks " << world_ranks << " on stream " << stream << " end...";
//         auto insertion = torch_nccl_comm_groups[stream_id + 1][device_id].insert(
//           {comm_group->world_ranks(), comm_group});
//         HT_ASSERT(insertion.second)
//           << "Failed to insert NCCLCommunicationGroup for ranks "
//           << comm_group->world_ranks() << ".";
//         it = insertion.first;
//       }
//     }
//     return it->second;
//   }
// }

// NCCLCommunicationGroup&
// NCCLCommunicationGroup::GetOrCreateWorldwide(const Stream& stream) {
//   HT_ASSERT(stream.device().is_cuda())
//     << "The argument \"stream\" for "
//     << "NCCLCommunicationGroup::GetOrCreate "
//     << "must be a CUDA stream. Got " << stream << ".";
//   // Note: stream id could be -1, we shall shift it by one when accessing
//   int stream_id = static_cast<int>(stream.stream_index());
//   int device_id = static_cast<int>(stream.device_index());

//   if (worldwide_torch_nccl_comm_groups[stream_id + 1][device_id].is_defined())
//     return worldwide_torch_nccl_comm_groups[stream_id + 1][device_id];
//   else
//     return GetOrCreate({}, stream);
// }


// } // namespace comm
// } // namespace impl
// } // namespace hetu
