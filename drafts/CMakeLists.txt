add_library(new_parser STATIC AST.cpp Parser.cpp Expr.cpp)

add_executable(draft_parser draft_parser.cpp)
target_link_libraries(draft_parser PRIVATE new_parser utils)