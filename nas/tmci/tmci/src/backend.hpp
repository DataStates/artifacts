#ifndef __TMCI_BACKEND_HPP
#define __TMCI_BACKEND_HPP
#include<vector>
#include <vector>
#include <memory>
#include <functional>
#include <tensorflow/core/framework/tensor.h>

namespace tmci {

template<typename T>
class BackendRegistry;

/**
 * @brief The Backend class is an abstract class representing an I/O backend
 * for TMCI. Concrete implementations must overload the Save and Load methods,
 * and have a constructor that takes a const char* null-terminated string
 * (the "config" parameter passed to the operation).
 */
class Backend {

    public:

    /**
     * @brief Constructor.
     */
    Backend();

    /**
     * @brief Destructor.
     */
    //virtual ~Backend();

    /**
     * @brief Saves the provided set of tensors. Implementations must return
     * 0 in case of success, or any other error code in case of failure.
     *
     * @param tensors Tensors to save.
     *
     * @return 0 in case of success, other error code in case of failure.
     */
    virtual int Save(const std::vector<std::reference_wrapper<const tensorflow::Tensor>>& tensors, std::vector<int32_t>&modelids_int32, std::vector<int32_t>&lids_int32) = 0;

    /**
     * @brief Loads the data into the provided tensors. Implementations must
     * return 0 in case of success, or any other error code in case of failure.
     *
     * @param tensors Tensors to load.
     *
     * @return 0 in case of success, other error code in case of failure.
     */
    virtual int Load(const std::vector<std::reference_wrapper<const tensorflow::Tensor>>& tensors, std::vector<int32_t>&modelids_int32, std::vector<int32_t>&lids_int32, std::vector<int32_t>&lowners) = 0;
    virtual bool store_meta(uint64_t id, uint64_t *edges, int m, uint64_t *lids, uint64_t *owners, uint64_t *sizes, int n, const float val_acc)=0; 
    virtual int get_composition(uint64_t id, uint64_t *lids, uint64_t *owners, int n)=0;
    virtual int get_prefix(uint64_t *edges, int n, uint64_t *id, uint64_t *result)=0;
    virtual bool update_ref_counter(uint64_t id, int value)=0;
    virtual int getTimeStamps(uint64_t *ts)=0;
    virtual int getTimeStampsByKey(uint64_t *ts, char *function_str)=0;
    virtual void clearTimeStamps()=0;
    virtual bool clearTimeStampsByKey(char *function_str)=0;
    virtual inline int getNumTimeStamps()=0;
    virtual inline int getNumTimeStampsByKey(char *function_str)=0;
    virtual inline int shutdown()=0;
    /**
     * @brief Factory function that creates an instance of backend given the name
     * of the backend and a configuration string.
     *
     * @param name Name of the backend.
     * @param config Configuration string for the backend.
     *
     * @return A unique_ptr to a Backend instance.
     */
    static std::shared_ptr<Backend> Create(const char* name, const char* config);

    private:

    template<typename T>
    friend class BackendRegistry;

    /**
     * @brief Registers a factory method for the given backend.
     *
     * @param name Name of the backend.
     * @param factory Factory function to create a Backend instance.
     */
    static void RegisterFactory(const char* name,
            std::function<std::shared_ptr<Backend>(const char*)>&& factory);
};

/**
 * @brief Helper class to help registration into the factory.
 *
 * @tparam T Backend type to register into the factory.
 */
template<typename T>
class BackendRegistry {

    public:

    BackendRegistry(const char* backend_name) {
        Backend::RegisterFactory(backend_name, [](const char* config) {
            return std::shared_ptr<T>(new T( config));
        });
    }
};

}

/**
 * \@def TMCI_REGISTER_BACKEND(name, class)
 * Registers the given class under the provided name.
 * The name should be a valid C qualifier.
 * Example:
 *  TMCI_REGISTER_BACKEND(dummy, DummyBackend);
 */
#define TMCI_REGISTER_BACKEND(__name, __class) \
    static tmci::BackendRegistry<__class> __tmci_registration_##__name(#__name);

#endif
