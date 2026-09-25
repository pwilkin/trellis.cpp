#include "uv_overlap_cover.h"
#include "retopo_process.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs=std::filesystem;

std::string read_text(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot read " + path.string());
    return {std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>()};
}

int run(const fs::path& program, const std::vector<std::string>& args,
        const fs::path& log, const fs::path& scratch) {
#ifdef _WIN32
    return trellis::retopo_process::run(program,args,
        {{L"TEMP",scratch.wstring()},{L"TMP",scratch.wstring()}},log);
#else
    const pid_t pid=fork();
    if (pid<0) throw std::runtime_error("Cannot start " + program.string());
    if (pid==0) {
        const int fd=open(log.c_str(),O_WRONLY|O_CREAT|O_TRUNC,0644);
        if (fd<0 || dup2(fd,STDOUT_FILENO)<0 || dup2(fd,STDERR_FILENO)<0) _exit(127);
        close(fd);
        if (setenv("TMPDIR",scratch.c_str(),1)!=0) _exit(127);
        std::vector<std::string> values{program.string()};
        values.insert(values.end(),args.begin(),args.end());
        std::vector<char*> raw;
        for (auto& value:values) raw.push_back(value.data());
        raw.push_back(nullptr);
        execv(program.c_str(),raw.data());
        _exit(127);
    }
    int status=0;
    if (waitpid(pid,&status,0)<0 || !WIFEXITED(status))
        throw std::runtime_error("Stage interrupted: " + program.string());
    return WEXITSTATUS(status);
#endif
}

bool set_env(const char* key,const char* value) {
#ifdef _WIN32
    return _putenv_s(key,value)==0 &&
           SetEnvironmentVariableW(trellis::retopo_process::wide(key).c_str(),
                                   trellis::retopo_process::wide(value).c_str())!=0;
#else
    return setenv(key,value,1)==0;
#endif
}

uint32_t face_count(const fs::path& uv) {
    std::ifstream input(uv,std::ios::binary);
    uint32_t counts[2]{};
    input.read(reinterpret_cast<char*>(counts),sizeof(counts));
    if (!input || !counts[0] || !counts[1]) throw std::runtime_error("Invalid UV header");
    return counts[1];
}

std::vector<uint32_t> read_ids(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot read selected UV faces");
    std::vector<uint32_t> ids;
    uint32_t value;
    while (input>>value) ids.push_back(value);
    if (!input.eof()) throw std::runtime_error("Invalid selected UV face list");
    return ids;
}

std::vector<uint32_t> select_repairs(const fs::path& quad_report,
        const fs::path& exact_report, uint32_t faces) {
    auto required=read_ids(quad_report.string()+".repair-faces.txt");
    std::ifstream tiny(exact_report.string()+".tiny.tsv");
    if (!tiny) throw std::runtime_error("Cannot read tiny UV faces");
    std::string line;
    std::getline(tiny,line);
    while (std::getline(tiny,line)) {
        std::istringstream row(line);
        uint32_t id;
        if (!(row>>id)) throw std::runtime_error("Invalid tiny UV face row");
        required.push_back(id);
    }
    std::ifstream pairs(exact_report.string()+".pairs.tsv");
    if (!pairs) throw std::runtime_error("Cannot read UV overlap pairs");
    std::vector<std::array<uint32_t,2>> overlaps;
    uint32_t a,b;
    while (pairs>>a>>b) overlaps.push_back({a,b});
    if (!pairs.eof()) throw std::runtime_error("Invalid UV overlap pair row");
    return trellis::greedy_uv_overlap_cover(faces,overlaps,required);
}

bool same_file_contents(const fs::path& a,const fs::path& b) {
    if (fs::file_size(a)!=fs::file_size(b)) return false;
    std::ifstream first(a,std::ios::binary),second(b,std::ios::binary);
    std::array<char,65536> x{},y{};
    while (first && second) {
        first.read(x.data(),x.size());second.read(y.data(),y.size());
        if (first.gcount()!=second.gcount() ||
            !std::equal(x.begin(),x.begin()+first.gcount(),y.begin())) return false;
    }
    return true;
}

bool same_post_geometry(const fs::path& a,const fs::path& b) {
    std::ifstream first(a,std::ios::binary),second(b,std::ios::binary);
    std::array<int32_t,4> x{},y{};
    first.read(reinterpret_cast<char*>(x.data()),sizeof(x));
    second.read(reinterpret_cast<char*>(y.data()),sizeof(y));
    if (!first || !second || x[0]<=0 || x[1]<=0 || x[2]<=0 || y[2]<=0 ||
        x[0]!=y[0] || x[1]!=y[1] || x[3]<=0 || y[3]<=0 ||
        fs::file_size(a)!=16+uint64_t(x[0])*12+uint64_t(x[1])*12+uint64_t(x[2])*36 ||
        fs::file_size(b)!=16+uint64_t(y[0])*12+uint64_t(y[1])*12+uint64_t(y[2])*36)
        return false;
    uint64_t remaining=uint64_t(x[0]+uint64_t(x[1]))*12;
    std::array<char,65536> v{},w{};
    while (remaining) {
        const auto n=std::streamsize(std::min<uint64_t>(remaining,v.size()));
        first.read(v.data(),n);second.read(w.data(),n);
        if (!first || !second || !std::equal(v.begin(),v.begin()+n,w.begin())) return false;
        remaining-=uint64_t(n);
    }
    return true;
}

double metric(const std::string& log,const std::string& key) {
    const auto at=log.find(key+"=");
    if (at==std::string::npos) throw std::runtime_error("Bake omitted " + key);
    return std::stod(log.substr(at+key.size()+1));
}

int main(int argc,char** argv) {
    const bool from_prepared=argc>1 && std::string(argv[1])=="--from-prepared";
    const bool from_post=argc>1 && std::string(argv[1])=="--from-post";
    if ((from_post && argc!=7 && argc!=8 && argc!=9) ||
        (from_prepared && argc!=5 && argc!=6) ||
        (!from_post && !from_prepared && argc!=4 && argc!=5)) {
        std::cerr<<"usage: trellis-retopo-atlas SOURCE.post QUADS.obj OUTPUT.glb [1024|2048|4096]\n"
                 <<"       trellis-retopo-atlas --from-prepared PREPARED.post SOURCE.post OUTPUT.glb [1024|2048|4096]\n"
                 <<"       trellis-retopo-atlas --from-post SOURCE.post OUTPUT.glb GRID FIRST_FACES FINAL_FACES [1024|2048|4096] [--no-weld-fill]\n";
        return 2;
    }
    try {
        const fs::path source=fs::absolute(from_post?argv[2]:(from_prepared?argv[3]:argv[1]));
        fs::path prepared,quads;
        if (from_prepared) prepared=fs::absolute(argv[2]);
        else if (!from_post) quads=fs::absolute(argv[2]);
        const fs::path output=fs::absolute(from_post?argv[3]:(from_prepared?argv[4]:argv[3]));
        const std::string resolution=from_post?(argc>=8?argv[7]:"4096"):
            (from_prepared?(argc==6?argv[5]:"4096"):(argc==5?argv[4]:"4096"));
        const auto parse_count=[](const char* token) {
            size_t used=0;
            const int value=std::stoi(token,&used);
            if (used!=std::string(token).size()) throw std::invalid_argument("Invalid preparation count");
            return value;
        };
        const bool raw_no_weld_fill=from_post && argc==9;
        if (raw_no_weld_fill && std::string(argv[8])!="--no-weld-fill")
            throw std::invalid_argument("Expected --no-weld-fill");
        const int grid=from_post?parse_count(argv[4]):0;
        const int first_faces=from_post?parse_count(argv[5]):0;
        const int final_faces=from_post?parse_count(argv[6]):0;
        if (from_post && (grid!=512 && grid!=1024 && grid!=1536 ||
                          first_faces<0 || final_faces<=0))
            throw std::invalid_argument("Invalid geometry preparation settings");
        if (resolution!="1024" && resolution!="2048" && resolution!="4096")
            throw std::runtime_error("Invalid atlas resolution");
        if (!fs::is_regular_file(source) ||
            (from_prepared && !fs::is_regular_file(prepared)) ||
            (!from_prepared && !from_post && !fs::is_regular_file(quads)))
            throw std::runtime_error("Missing source POST, prepared POST, or quad OBJ");
        fs::path bake_source=source;
        if (const char* alternate=std::getenv("TRELLIS_RETOPO_PBR_POST")) {
            bake_source=fs::absolute(alternate);
            if (!fs::is_regular_file(bake_source) || !same_post_geometry(source,bake_source))
                throw std::runtime_error("Retopo PBR POST geometry differs from source POST");
        }
        const fs::path scratch=output.parent_path();
        fs::create_directories(scratch);
        if (!set_env("TRELLIS_PROBE_ATLAS_RES",resolution.c_str()) ||
            !set_env("TRELLIS_PROBE_ATLAS_PADDING","2") ||
            (!std::getenv("TRELLIS_PROBE_ATLAS_TPU") &&
             !set_env("TRELLIS_PROBE_ATLAS_TPU","0.0039")) ||
            (!std::getenv("TRELLIS_PROBE_ATLAS_BACKOFF") &&
             !set_env("TRELLIS_PROBE_ATLAS_BACKOFF","0.95")))
            throw std::runtime_error("Cannot configure native atlas resolution");
        const fs::path prefix=scratch/(output.stem().string()+".retopo");
        const fs::path accepted_report=prefix.string()+"-accepted.json";
        fs::remove(accepted_report);
#ifdef _WIN32
        const fs::path binaries=trellis::retopo_process::executable_directory();
#else
        const fs::path binaries=fs::canonical("/proc/self/exe").parent_path();
#endif
        const auto stage=[&](const char* name,const std::vector<std::string>& args,
                             const fs::path& log,bool allow_one=false) {
            const int code=run(binaries/name,args,log,scratch);
            if (code!=0 && !(allow_one && code==1))
                throw std::runtime_error(std::string(name)+" failed; see "+log.string());
            return code;
        };
        if (from_post) {
            if (!set_env("TRELLIS_QEM_PRESERVE_TOPOLOGY","1") ||
                !set_env("TRELLIS_QEM_COLLISION_GUARD","1") ||
                !set_env("TRELLIS_REMESH_TETS","1") ||
                !set_env("TRELLIS_TETS_MIN_EDGE_FRACTION","0.05"))
                throw std::runtime_error("Cannot configure guarded geometry preparation");
            const fs::path initial=prefix.string()+"-initial.post";
            prepared=prefix.string()+"-prepared.post";
            std::vector<std::string> first{source.string(),prefix.string()+"-initial-unused.glb",
                "--geometry-res",std::to_string(grid),"--keep-components","--no-bake",
                "--dump-geometry-post",initial.string()};
            if (raw_no_weld_fill) first.insert(first.end(),{"--no-weld","--no-fill"});
            if (first_faces==0) first.insert(first.end(),{"--decim","0"});
            else first.insert(first.end(),{"--faces",std::to_string(first_faces)});
            stage("post-replay",first,prefix.string()+"-prepare-initial.log");
            stage("post-replay",{initial.string(),prefix.string()+"-prepared-unused.glb",
                  "--no-remesh","--no-weld","--no-fill","--keep-components",
                  "--faces",std::to_string(final_faces),"--no-bake",
                  "--dump-geometry-post",prepared.string()},
                  prefix.string()+"-prepare-final.log");
            if (!fs::is_regular_file(prepared) || fs::file_size(prepared)==0)
                throw std::runtime_error("Geometry preparation produced no POST");
        }
        if (from_prepared || from_post) {
            quads=prefix.string()+"-covered.obj";
            stage("post-replay",{prepared.string(),prefix.string()+"-unused.glb",
                  "--no-remesh","--no-weld","--no-fill","--decim","0","--no-bake",
                  "--shape-flips","--shape-smooth","--shape-min-angle","2",
                  "--shape-max-move","0.00025",
                  "--quad-safe-output",prefix.string()+"-mixed.obj",
                  "--quad-refine-output",prefix.string()+"-refined.obj",
                  "--quad-refine-qualified","--quad-repair-source",source.string(),
                  "--quad-source-cover-output",quads.string()},prefix.string()+"-geometry.log");
            if (!fs::is_regular_file(quads) || fs::file_size(quads)==0)
                throw std::runtime_error("Qualified geometry stage produced no quad mesh");
        }
        const fs::path topology_report=prefix.string()+"-topology.json";
        stage("trellis-audit-quad-topology",
              {quads.string(),topology_report.string()},prefix.string()+"-topology.log");
        const std::string topology=read_text(topology_report);
        if (topology.find("\"accepted\":true")==std::string::npos)
            throw std::runtime_error("Quad topology auditor did not accept the mesh");
        double forward_bound=0,reverse_bound=0;
        if (from_prepared || from_post) {
            const auto geometry=read_text(prefix.string()+"-geometry.log");
            forward_bound=metric(geometry,"upper_bound");
            reverse_bound=metric(geometry,"reverse_bound");
            if (!std::isfinite(forward_bound) || !std::isfinite(reverse_bound) ||
                forward_bound>.008 || reverse_bound>.008)
                throw std::runtime_error("Source-cover certificate exceeded distance gate");
        }
        fs::path uv=prefix.string()+"-chart.uv.bin";
        stage("trellis-chart-quads",{quads.string(),uv.string(),resolution},
              prefix.string()+"-chart.log");
        int accepted_round=-1;
        for (int round=0;round<=16;++round) {
            const fs::path quad_report=prefix.string()+"-round"+std::to_string(round)+".quad.json";
            const fs::path exact_report=prefix.string()+"-round"+std::to_string(round)+".exact.json";
            stage("trellis-audit-quad-uv",{quads.string(),uv.string(),quad_report.string()},
                  prefix.string()+"-round"+std::to_string(round)+"-quad.log");
            stage("trellis-audit-uv",{uv.string(),exact_report.string()},
                  prefix.string()+"-round"+std::to_string(round)+"-exact.log",true);
            const bool quad_ok=read_text(quad_report).find("\"four_corner_uv_export_possible\":true")!=std::string::npos;
            const bool exact_ok=read_text(exact_report).find("\"accepted\":true")!=std::string::npos;
            if (quad_ok && exact_ok) { accepted_round=round;break; }
            if (round==16) break;
            const auto selection=select_repairs(quad_report,exact_report,face_count(uv));
            if (selection.empty()) throw std::runtime_error("UV defects have no repair cover");
            const fs::path selected=prefix.string()+"-round"+std::to_string(round)+".selected.txt";
            std::ofstream ids(selected);
            for (uint32_t id:selection) ids<<id<<'\n';
            ids.close();
            if (!ids) throw std::runtime_error("Cannot write UV repair selection");
            const fs::path next=prefix.string()+"-round"+std::to_string(round+1)+".uv.bin";
            stage("trellis-repack-uv",{uv.string(),next.string(),selected.string(),quads.string()},
                  prefix.string()+"-round"+std::to_string(round)+"-repack.log");
            uv=next;
            std::cout<<"UV repair round "<<round<<" selected="<<selection.size()<<'\n';
        }
        if (accepted_round<0) throw std::runtime_error("UV atlas did not pass within 16 repairs");
        stage("trellis-bake-quads",{bake_source.string(),quads.string(),output.string(),
              resolution,"packed",uv.string()},prefix.string()+"-bake.log");
        const auto bake=read_text(prefix.string()+"-bake.log");
        const double missing=metric(bake,"missing_voxel_texels");
        const double over=metric(bake,"projected_samples_above_0_008");
        const double max_distance=metric(bake,"maximum_source_distance");
        if (missing!=0 || over!=0 || !std::isfinite(max_distance) || max_distance>.008)
            throw std::runtime_error("PBR bake failed source-coverage or projection gate");
        if (!same_file_contents(uv,output.string()+".uv.bin"))
            throw std::runtime_error("Exported UV mesh differs from accepted atlas");
        const fs::path pending_report=accepted_report.string()+".pending";
        std::ofstream report(pending_report);
        report<<"{\"atlas\":"<<resolution<<",\"repair_rounds\":"<<accepted_round
              <<",\"missing_voxel_texels\":0,\"projected_samples_above_0_008\":0"
              <<",\"maximum_source_distance\":"<<max_distance
              <<",\"atlas_exact_in_export\":true,\"topology_accepted\":true"
              <<",\"topology\":"<<topology
              <<",\"source_cover_forward_bound\":"<<forward_bound
              <<",\"source_cover_reverse_bound\":"<<reverse_bound
              <<",\"geometry_qualified_from_prepared\":"
              <<(from_prepared || from_post?"true":"false")
              <<",\"raw_post_prepared\":" << (from_post?"true":"false")
              <<",\"alternate_pbr_post\":" << (bake_source!=source?"true":"false")
              <<",\"raw_no_weld_fill\":" << (raw_no_weld_fill?"true":"false")
              << "}\n";
        report.close();
        if (!report) throw std::runtime_error("Cannot write accepted retopology report");
        fs::rename(pending_report,accepted_report);
        std::cout<<"Accepted textured quad export "<<output<<" after "<<accepted_round
                 <<" UV repairs; maximum source projection "<<max_distance<<'\n';
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<'\n';
        return 1;
    }
}
