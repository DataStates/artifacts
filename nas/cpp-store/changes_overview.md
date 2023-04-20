# High Level Changes



# transfer_methods.py
### extract_tensor_config() : 
Replaced  `_extract_tensor()` with `_extract_tensor_config()`. The new method does not extracts tensors (i.e doesn't call `get_weights()`) because we directly get the pointers via tmci. Currently, we use this method to extract tensor sizes and create a list of tensor ids (`lids`). 

### get_composition: 

 - In the old version (before I touched any code) - `self.lib.get_composition(cid, prefix_lids, lowners, len(prefix_lids))` was NOT called in transfer_methods.transfer(). This resulted in a bug  because in transfer_methods.store() , we were accessing composition
   info via comp dictionary but it was never getting updated. To fix it, I simply called `self.lib.get_composition(z` in  `transfer_methods.transfer().` It was already defined in cpp-store ctypes interface, i just had to call it.  
 -  In the old version of cpp-store: in `model_client::read_layers()` - 
   `get_composition()` was called. This resulted in a redundancy after I
   made the change above. That is: get_composition() is now called both in
   `transfer_methods.transfer()` and also in `model_client::read_layers()`.
   So, I removed the `get_composition()` call from
   `model_client::read_layers()`, and passed the composition info ( `lowners`) from transfer_methods to cpp-store (see/track `lowners` param in `tmci.checkpoint.load_weights`).

- 

# transfer_methods - cpp-store Interface

Previously: client_lib.cpp contained all the methods that interfaced cpp-store with transfer_methods.py

- Now: DummyBackend.cpp has the interfaces. 
- `class DummyBackend `inherits from `tmci::backend` (We needed to design it this way to overload Save and Load from TMCI)
- All the other methods (except for Save , Load and timestamp related methods) are exactly the same as they were in client_lib.cpp
- In DummyBackend.cpp - each class method that we expose to transfer_methods.py, has a corresponding C wrapper. The wrapper just gets a pointer to DummyBackend object by calling *`tmci::Backend::Create(backend, config)` - defined in tmci). Using this pointer it can then instantiate the corresponding class methods.
- TLDR call chain: transfer_methods.lib.XYZ -> extern "C" abc -> DummyBackend::abc
*Note   : backend and config (args to tmci::create) uniquely identify a DummyBackend object, look at tmci factory methods (in tmci/src/backend.cpp and tmci/src/backend.hpp) for more info. 

# Read and Load Tensors - zero copy when tensors are on the CPU

- In the previous version, we were copying the tensors to a vector of strings (one string per tensor) in DummyBackend::Save and DummyBackend::Load. Now, we only copy the tensors (to the host memory) if they are GPU resident . Otherwise, we pass the pointers directly from DummyBackend to model::client. Note, I changed the API of `model_client::read_layers`  and `model_client::store_layers` so that it directly accepts a std::vector of segments.

# Timestamps
- All timestamps are measured on the model_server and the clocks begin/end in model_server methods. 
- However, the timestamps ( `std::map<std::string, std::vector<uint64_t>>` )are stored as an attribute in the DummyBackend class. The keys are the function names and the values are a vector of timestamps
- DummyBackend passes to the client (which then passes it as an RPC arg to the server) the timestamps.
- The model_server methods/RPCs return a std::pair or std::vector of timestamps to the client and consequently gets updated in  DummyBackend .
- In transfer_methods - you can call get_time_stamps() and clear_time_stamps(). If no args are passed, it returns/deletes the entire map. You can optionally provide a key (string of function whose timestamps you'd like to retrieve/delete).

gen_models_with_pop.py
# Misc Note
The server writes it's addresss to a hard coded file, and the client reads from the file. You'll either haave to chcange the path (since currently it is hard-coded to my home directory) or change it back to what it was before.
