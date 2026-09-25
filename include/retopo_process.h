#pragma once

#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <cwchar>
#include <cwctype>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trellis::retopo_process {
namespace fs = std::filesystem;

inline fs::path executable_directory() {
    std::wstring buffer(MAX_PATH, L'\0');
    for (;;) {
        const DWORD length=GetModuleFileNameW(nullptr,buffer.data(),static_cast<DWORD>(buffer.size()));
        if (!length) throw std::runtime_error("Cannot locate executable");
        if (length<buffer.size()) {
            buffer.resize(length);
            return fs::path(buffer).parent_path();
        }
        buffer.resize(buffer.size()*2);
    }
}

inline std::wstring wide(const std::string& text) {
    if (text.empty()) return {};
    const int length=MultiByteToWideChar(CP_ACP,0,text.data(),
                                         static_cast<int>(text.size()),nullptr,0);
    if (!length) throw std::runtime_error("Invalid process argument");
    std::wstring result(length,L'\0');
    MultiByteToWideChar(CP_ACP,0,text.data(),
                        static_cast<int>(text.size()),result.data(),length);
    return result;
}

inline std::wstring quote(const std::wstring& argument) {
    std::wstring result=L"\"";
    size_t slashes=0;
    for (wchar_t ch:argument) {
        if (ch==L'\\') { ++slashes;continue; }
        if (ch==L'"') {
            result.append(slashes*2+1,L'\\');
            result.push_back(ch);
        } else {
            result.append(slashes,L'\\');
            result.push_back(ch);
        }
        slashes=0;
    }
    result.append(slashes*2,L'\\');
    return result+L'"';
}

inline std::vector<wchar_t> environment(
    const std::vector<std::pair<std::wstring,std::wstring>>& overrides) {
    LPWCH inherited=GetEnvironmentStringsW();
    if (!inherited) throw std::runtime_error("Cannot read process environment");
    std::vector<std::wstring> entries;
    for (const wchar_t* p=inherited;*p;p+=wcslen(p)+1) entries.emplace_back(p);
    FreeEnvironmentStringsW(inherited);
    for (const auto& [key,value]:overrides) {
        if (key.empty() || key.find(L'=')!=std::wstring::npos ||
            key.find(L'\0')!=std::wstring::npos || value.find(L'\0')!=std::wstring::npos)
            throw std::runtime_error("Invalid process environment override");
        entries.erase(std::remove_if(entries.begin(),entries.end(),[&](const std::wstring& entry) {
            const auto equal=entry.find(L'=',!entry.empty() && entry[0]==L'='?1:0);
            return equal!=std::wstring::npos && equal==key.size() &&
                _wcsnicmp(entry.c_str(),key.c_str(),key.size())==0;
        }),entries.end());
        if (!value.empty()) entries.push_back(key+L'='+value);
    }
    std::sort(entries.begin(),entries.end(),[](const std::wstring& a,const std::wstring& b) {
        return _wcsicmp(a.c_str(),b.c_str())<0;
    });
    std::vector<wchar_t> block;
    for (const auto& entry:entries) {
        block.insert(block.end(),entry.begin(),entry.end());
        block.push_back(L'\0');
    }
    block.push_back(L'\0');
    if (entries.empty()) block.push_back(L'\0');
    return block;
}

inline int run(const fs::path& program,const std::vector<std::string>& arguments,
               const std::vector<std::pair<std::wstring,std::wstring>>& overrides={},
               const fs::path& log={}) {
    const fs::path executable=program.extension().empty()?fs::path(program.wstring()+L".exe"):program;
    std::wstring command=quote(executable.wstring());
    for (const auto& argument:arguments) command+=L' '+quote(wide(argument));
    std::vector<wchar_t> mutable_command(command.begin(),command.end());
    mutable_command.push_back(L'\0');
    auto env=environment(overrides);
    HANDLE output=INVALID_HANDLE_VALUE;
    STARTUPINFOW startup{};
    startup.cb=sizeof(startup);
    BOOL inherit=FALSE;
    if (!log.empty()) {
        SECURITY_ATTRIBUTES security{sizeof(security),nullptr,TRUE};
        output=CreateFileW(log.c_str(),GENERIC_WRITE,FILE_SHARE_READ,&security,
                           CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
        if (output==INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot open stage log");
        startup.dwFlags=STARTF_USESTDHANDLES;
        startup.hStdInput=GetStdHandle(STD_INPUT_HANDLE);
        startup.hStdOutput=output;
        startup.hStdError=output;
        inherit=TRUE;
    }
    PROCESS_INFORMATION child{};
    const BOOL launched=CreateProcessW(executable.c_str(),mutable_command.data(),nullptr,
        nullptr,inherit,CREATE_UNICODE_ENVIRONMENT,env.data(),nullptr,&startup,&child);
    if (output!=INVALID_HANDLE_VALUE) CloseHandle(output);
    if (!launched) throw std::runtime_error("Cannot launch " + executable.string() +
                                            " (Windows error " + std::to_string(GetLastError()) + ")");
    const DWORD waited=WaitForSingleObject(child.hProcess,INFINITE);
    DWORD code=1;
    const BOOL exited=waited==WAIT_OBJECT_0 && GetExitCodeProcess(child.hProcess,&code);
    CloseHandle(child.hThread);
    CloseHandle(child.hProcess);
    if (!exited) throw std::runtime_error("Stage interrupted: " + executable.string());
    return static_cast<int>(code);
}
}

#endif
