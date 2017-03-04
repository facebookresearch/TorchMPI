local MPI = require("torchmpi.env")
local ffi = require("ffi")
local types = require("torchmpi.types")

local function declMPI(withCuda)
   local allreduce_def, broadcast_def, reduce_def, sendreceive_def, parameterserver_def = "", "", "", "", ""
   for _, v in pairs(types.torch) do
      allreduce_def = allreduce_def .. [[
         void torchmpi_allreduce_TH]] .. v .. [[Tensor(TH]] .. v .. [[Tensor* input, TH]] .. v .. [[Tensor* output);
         SynchronizationHandle* torchmpi_async_allreduce_TH]] .. v .. [[Tensor(TH]] .. v .. [[Tensor* input, TH]] .. v .. [[Tensor* output);
         void torchmpi_p2p_allreduce_TH]] .. v .. [[Tensor(TH]] .. v .. [[Tensor* input, TH]] .. v .. [[Tensor* output);
         SynchronizationHandle* torchmpi_async_p2p_allreduce_TH]] .. v .. [[Tensor(TH]] .. v .. [[Tensor* input, TH]] .. v .. [[Tensor* output);
      ]]
      broadcast_def = broadcast_def .. [[
         void torchmpi_broadcast_TH]] .. v .. [[Tensor(TH]] .. v .. [[Tensor* input, int src);
         SynchronizationHandle* torchmpi_async_broadcast_TH]] .. v .. [[Tensor(TH]] .. v .. [[Tensor* input, int src);
      ]]
      reduce_def = reduce_def .. [[
         void torchmpi_reduce_TH]] .. v .. [[Tensor(TH]] .. v .. [[Tensor* input, TH]] .. v .. [[Tensor* output, int dst);
         SynchronizationHandle* torchmpi_async_reduce_TH]] .. v .. [[Tensor(TH]] .. v .. [[Tensor* input, TH]] .. v .. [[Tensor* output, int dst);
      ]]
      sendreceive_def = sendreceive_def .. [[
         void torchmpi_sendreceive_TH]] .. v .. [[Tensor(TH]] .. v .. [[Tensor* input, int src, int dst);
      ]]
      parameterserver_def = parameterserver_def .. [[
         void* torchmpi_parameterserver_init_TH]] .. v .. [[Tensor(TH]] .. v .. [[Tensor* input);
      ]] .. [[
         ParameterServerSynchronizationHandle* torchmpi_parameterserver_send_TH]] .. v .. [[Tensor(void* PS, TH]] .. v .. [[Tensor* input, const char* updateRuleName);
      ]] .. [[
         ParameterServerSynchronizationHandle* torchmpi_parameterserver_receive_TH]] .. v .. [[Tensor(void* PS, TH]] .. v .. [[Tensor* input);
      ]] .. [[
         ParameterServerSynchronizationHandle* torchmpi_parameterserver_synchronize_handle(ParameterServerSynchronizationHandle*);
         void torchmpi_parameterserver_free(void* ps);
         void torchmpi_parameterserver_free_all();
      ]]
   end

   local allreduce_scalar_def, broadcast_scalar_def, reduce_scalar_def, sendreceive_scalar_def = "", "", "", ""
   for _, v in pairs(types.C) do
      allreduce_scalar_def = allreduce_scalar_def .. [[
         ]] .. v .. [[ torchmpi_allreduce_]] .. v .. [[(]] .. v .. [[ val);
      ]]
      broadcast_scalar_def = broadcast_scalar_def .. [[
         ]] .. v .. [[ torchmpi_broadcast_]] .. v .. [[(]] .. v .. [[ val, int src);
      ]]
      reduce_scalar_def = reduce_scalar_def .. [[
         ]] .. v .. [[ torchmpi_reduce_]] .. v .. [[(]] .. v .. [[ val, int dst);
      ]]
      sendreceive_scalar_def = sendreceive_scalar_def .. [[
         ]] .. v .. [[ torchmpi_sendreceive_]] .. v .. [[(]] .. v .. [[ val, int src, int dst);
      ]]
   end

   local def = [[
      typedef void* cudaStream_t;
      typedef struct SynchronizationHandle {
         bool hasMPIRequest;
         bool hasFuture;
         bool hasStream;
         cudaStream_t stream;
         size_t mpiRequestIndex;
         size_t futureIndex;
      } SynchronizationHandle;

      typedef struct ParameterServerSynchronizationHandle {
        bool hasFuture;
        size_t futureIndex;
      } ParameterServerSynchronizationHandle;

   ]]
   if withCuda then
      require('cutorch')
      for _, v in pairs(types.torch) do
         -- THCudaTensor instead of THCudaFloatTensor sigh
         if v == 'Float' then v = '' end
         allreduce_def = allreduce_def .. [[
            void torchmpi_allreduce_THCuda]] .. v .. [[Tensor]] ..
               [[(THCState* state, THCuda]] .. v .. [[Tensor* input, THCuda]] .. v .. [[Tensor* output);
            SynchronizationHandle* torchmpi_async_allreduce_THCuda]] .. v .. [[Tensor]] ..
               [[(THCState* state, THCuda]] .. v .. [[Tensor* input, THCuda]] .. v .. [[Tensor* output);
            void torchmpi_p2p_allreduce_THCuda]] .. v .. [[Tensor]] ..
               [[(THCState* state, THCuda]] .. v .. [[Tensor* input, THCuda]] .. v .. [[Tensor* output);
            SynchronizationHandle* torchmpi_async_p2p_allreduce_THCuda]] .. v .. [[Tensor]] ..
               [[(THCState* state, THCuda]] .. v .. [[Tensor* input, THCuda]] .. v .. [[Tensor* output);
            void torchmpi_nccl_allreduce_THCuda]] .. v .. [[Tensor]] ..
               [[(THCState* state, THCuda]] .. v .. [[Tensor* input, THCuda]] .. v .. [[Tensor* output);
            SynchronizationHandle* torchmpi_async_nccl_allreduce_THCuda]] .. v .. [[Tensor]] ..
               [[(THCState* state, THCuda]] .. v .. [[Tensor* input, THCuda]] .. v .. [[Tensor* output);
         ]]
         broadcast_def = broadcast_def .. [[
            void torchmpi_broadcast_THCuda]] .. v .. [[Tensor]] ..
               [[(THCState* state, THCuda]] .. v .. [[Tensor* input, int src);
            SynchronizationHandle* torchmpi_async_broadcast_THCuda]] .. v .. [[Tensor]] ..
               [[(THCState* state, THCuda]] .. v .. [[Tensor* input, int src);
            void torchmpi_p2p_broadcast_THCuda]] .. v .. [[Tensor]] ..
               [[(THCState* state, THCuda]] .. v .. [[Tensor* input, int src);
            SynchronizationHandle* torchmpi_async_p2p_broadcast_THCuda]] .. v .. [[Tensor]] ..
               [[(THCState* state, THCuda]] .. v .. [[Tensor* input, int src);
            void torchmpi_nccl_broadcast_THCuda]] .. v .. [[Tensor]] ..
               [[(THCState* state, THCuda]] .. v .. [[Tensor* input, int src);
            SynchronizationHandle* torchmpi_async_nccl_broadcast_THCuda]] .. v .. [[Tensor]] ..
               [[(THCState* state, THCuda]] .. v .. [[Tensor* input, int src);
         ]]
         reduce_def = reduce_def .. [[
            void torchmpi_reduce_THCuda]] .. v .. [[Tensor]] ..
               [[(THCState* state, THCuda]] .. v .. [[Tensor* input, THCuda]] .. v .. [[Tensor* output, int dst);
            void torchmpi_nccl_reduce_THCuda]] .. v .. [[Tensor]] ..
               [[(THCState* state, THCuda]] .. v .. [[Tensor* input, THCuda]] .. v .. [[Tensor* output, int dst);
            SynchronizationHandle* torchmpi_async_nccl_reduce_THCuda]] .. v .. [[Tensor]] ..
               [[(THCState* state, THCuda]] .. v .. [[Tensor* input, THCuda]] .. v .. [[Tensor* output, int dst);
         ]]
         sendreceive_def = sendreceive_def .. [[
            void torchmpi_sendreceive_THCuda]] .. v .. [[Tensor]] ..
               [[(THCState* state, THCuda]] .. v .. [[Tensor* input, int src, int dst);
         ]]
      end
   end

   def = def .. allreduce_def .. broadcast_def .. reduce_def .. sendreceive_def .. parameterserver_def ..
      allreduce_scalar_def .. broadcast_scalar_def .. reduce_scalar_def .. sendreceive_scalar_def ..
   [[
      void torchmpi_start();
      void torchmpi_stop();
      int torchmpi_rank();
      int torchmpi_size();
      int torchmpi_has_nccl();
      void torchmpi_free_ipc_descriptors();
      const char* torchmpi_communicator_names();
      int torchmpi_push_communicator(const char* key);
      void torchmpi_set_communicator(int level);
      void torchmpi_set_collective_span(int begin, int end);
      int torchmpi_num_nodes_in_communicator();
      void torchmpi_free_ipc_descriptors();
      void torchmpi_set_flat_collectives();
      void torchmpi_set_hierarchical_collectives();
      void torchmpi_set_staged_collectives();
      void torchmpi_set_direct_collectives();
      void torchmpi_set_small_cpu_broadcast_size(int n); // default 1 << 13
      void torchmpi_set_small_cpu_allreduce_size(int n); // default 1 << 16
      void torchmpi_set_small_gpu_broadcast_size(int n); // default 1 << 13
      void torchmpi_set_small_gpu_allreduce_size(int n); // default 1 << 16
      int torchmpi_get_small_cpu_broadcast_size();
      int torchmpi_get_small_cpu_allreduce_size();
      int torchmpi_get_small_gpu_broadcast_size();
      int torchmpi_get_small_gpu_allreduce_size();
      void torchmpi_set_num_buffers_per_cpu_collective(int n); // default 1
      void torchmpi_set_num_buffers_per_gpu_collective(int n); // default 1
      int torchmpi_get_num_buffers_per_cpu_collective();
      int torchmpi_get_num_buffers_per_gpu_collective();
      void torchmpi_set_collective_num_threads(int n);           // default 4
      void torchmpi_set_collective_thread_pool_size(int n);      // default 1 << 20
      void torchmpi_set_parameterserver_num_threads(int n);      // default 4
      void torchmpi_set_parameterserver_thread_pool_size(int n); // default 1 << 20
      int torchmpi_get_collective_num_threads();
      int torchmpi_get_collective_thread_pool_size();
      int torchmpi_get_parameterserver_num_threads();
      int torchmpi_get_parameterserver_thread_pool_size();
      SynchronizationHandle* torchmpi_synchronize_handle(SynchronizationHandle* h);
      void torchmpi_barrier();
      void customBarrier();
   ]]

   ffi.cdef(def)
   MPI.C = ffi.load(assert(package.searchpath('libtorchmpi', package.cpath), 'libtorchmpi not found'), true)
end


return declMPI
