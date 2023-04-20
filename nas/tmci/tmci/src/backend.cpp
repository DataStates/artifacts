#include "backend.hpp"
#include <unordered_map>

namespace tmci {



    std::unordered_map<std::string, std::function<std::shared_ptr<Backend>(const char*)>> _backend_factories;
    std::unordered_map<std::string, std::shared_ptr<Backend> > _pointers;
    //std::unordered_map<std::string, std::function<std::unique_ptr<Backend>(const char*, std::vector<int>&lids)>> _backend_factories;

    Backend::Backend() {}

    //Backend::~Backend() {}

    std::shared_ptr<Backend> Backend::Create(const char* name, const char* config) {
        std::string name_str = std::string(name);
        std::string key = std::string(name) + "#" + std::string(config);
        if(_pointers.count(key) == 0) {
            if(_backend_factories.count(name) == 0){

                throw std::invalid_argument(std::string("TMCI backend \"") + name + "\" not found");
            }
            _pointers[key] = _backend_factories[name](config);
        }
        return _pointers[key];
    }

    void Backend::RegisterFactory(const char* name, 
            std::function<std::shared_ptr<Backend>(const char*)>&& factory) {
        if(_backend_factories.count(name) != 0)
            throw std::runtime_error(std::string("TMCI backend \"") + name + "\" already registered");
        _backend_factories[name] = std::move(factory);
    }


}
