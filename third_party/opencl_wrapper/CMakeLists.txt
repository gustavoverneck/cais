# Cria a biblioteca chamada "cais"
add_library(cais
    solver.cpp
    cpu_engine.cpp
    opencl_engine.cpp
)

# Define que os headers públicos estão na pasta "include" do projeto
target_include_directories(cais PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

# Linka a biblioteca com suas dependências (Eigen e OpenCL)
target_link_libraries(cais PUBLIC
    Eigen3::Eigen
    OpenCL::OpenCL
)

# Adiciona um alias para que os usuários possam linkar com cais::cais
add_library(cais::cais ALIAS cais)