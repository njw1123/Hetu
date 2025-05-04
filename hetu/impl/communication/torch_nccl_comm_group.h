// #pragma once

// #include "hetu/impl/communication/comm_group.h"
// #include "hetu/core/ndarray.h"
// #include "hetu/impl/kernel/TorchUtils.h"
// #include "hetu/impl/utils/cuda_utils.h" // 用于 CUDA stream 操作

// #include <torch/torch.h>
// #include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
// #include <torch/csrc/distributed/c10d/Store.hpp>
// #include <torch/csrc/distributed/c10d/FileStore.hpp> 
// #include <torch/csrc/distributed/c10d/Types.hpp>
// #include <torch/csrc/distributed/c10d/Work.hpp>
// #include <c10/cuda/CUDAStream.h> 

// #include <vector>
// #include <string>
// #include <map>

// namespace hetu {
// namespace impl {
// namespace comm {

// // 前向声明
// class NCCLCommunicationGroupDef;
// class NCCLCommunicationGroup;


// // Torch NCCL 通信组定义类
// class NCCLCommunicationGroupDef : public CommunicationGroupDef {
//  protected:
//   friend class NCCLCommunicationGroup;
//   struct constructor_access_key {};

//   NCCLCommunicationGroupDef(const std::vector<int>& world_ranks,
//                                  const Stream& stream);

//  public:
//   NCCLCommunicationGroupDef(const constructor_access_key&,
//                                  const std::vector<int>& world_ranks,
//                                  const Stream& stream)
//   : NCCLCommunicationGroupDef(world_ranks, stream) {}

//   ~NCCLCommunicationGroupDef();


//   void Broadcast(NDArray& data, int broadcaster_world_rank) override;

//   void AllReduce(const NDArray& input, NDArray& output,
//                  ReductionType red_type = kSUM) override;

//   void AllReduceCoalesce(const NDArrayList& inputs, NDArrayList& outputs,
//                          NDArray contiguous_buffers,
//                          ReductionType red_type = kSUM) override;

//   void AlltoAll(const NDArray& input, NDArray& output) override;

//   void Reduce(const NDArray& input, NDArray& output, int reducer_world_rank,
//               ReductionType red_type = kSUM) override;

//   void AllGather(const NDArray& input, NDArray& output, int32_t gather_dim = 0) override;

//   void ReduceScatter(const NDArray& input, NDArray& output,
//                      int32_t scatter_dim = 0, ReductionType red_type = kSUM) override;

//   void Gather(const NDArray& input, NDArray& output, int gatherer_world_rank) override;

//   void Scatter(const NDArray& input, NDArray& output, int scatterer_world_rank) override;

//   void Send(const NDArray& data, int receiver_world_rank) override;

//   void Recv(NDArray& data, int sender_world_rank) override;

//   CommTask ISend(const NDArray& data, int receiver_world_rank) override;

//   CommTask IRecv(NDArray& data, int sender_world_rank) override;

//   void BatchedISendIRecv(const std::vector<CommTask>& tasks) override;

//   void Barrier(bool sync = false) override;

//   void Sync() override;

//   std::string backend() const override {
//     return "TorchNCCL";
//   }

//  protected:

//   // --- 成员变量 ---
//   c10::intrusive_ptr<c10d::Store> _store; // 用于 rank 间协调的存储
//   c10::intrusive_ptr<c10d::ProcessGroupNCCL> _process_group; // Torch NCCL 进程组实例
//   // 存储待处理的异步工作项，用于 Sync()
//   std::vector<c10::intrusive_ptr<c10d::Work>> _pending_work;
// };


// // Torch NCCL 通信组包装类 (最终用户使用的类)
// class NCCLCommunicationGroup final
// : public CommGroupWrapper<NCCLCommunicationGroupDef> {
//  protected:
//   // 保护构造函数，强制使用工厂方法
//   NCCLCommunicationGroup(const std::vector<int>& world_ranks,
//                               const Stream& stream)
//   : CommGroupWrapper<NCCLCommunicationGroupDef>(
//       make_ptr<NCCLCommunicationGroupDef>(
//         NCCLCommunicationGroupDef::constructor_access_key(), world_ranks,
//         stream)) {}

//  public:
//   NCCLCommunicationGroup() = default;

//   // --- 静态工厂方法 (GetOrCreate 模式) ---
//   // 根据 world_ranks 和 stream 获取或创建通信组
//   static NCCLCommunicationGroup&
//   GetOrCreate(const std::vector<int>& world_ranks, const Stream& stream);

//   // 根据 world_ranks 和 device 获取或创建通信组 (使用默认 stream 类型)
//   static NCCLCommunicationGroup&
//   GetOrCreate(const std::vector<int>& world_ranks, Device device) {
//     return GetOrCreate(
//       world_ranks,
//       Stream(device, world_ranks.size() != 2 ? kCollectiveStream : kP2PStream));
//   }

//   // 获取或创建全局通信组 (包含所有 world ranks)
//   static NCCLCommunicationGroup& GetOrCreateWorldwide(const Stream& stream);

//   // 获取或创建全局通信组 (使用默认 stream 类型)
//   static NCCLCommunicationGroup& GetOrCreateWorldwide(Device device) {
//     return GetOrCreateWorldwide(Stream(device, kCollectiveStream));
//   }
// };

// } // namespace comm
// } // namespace impl
// } // namespace hetu
