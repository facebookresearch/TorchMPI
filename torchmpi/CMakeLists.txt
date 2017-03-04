file(GLOB luafiles RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.lua")
install(FILES ${luafiles} DESTINATION ${LUA_PATH}/${PKGNAME})
