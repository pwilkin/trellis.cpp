// trellis-server — resident HTTP wrapper around the TRELLIS.2 image->3D pipeline.
//
//   GET  /health     -> "ok"
//   POST /generate    multipart/form-data with an "image" file part; optional text
//                      fields "seed", "resolution" (512/1024/1536), "bg_removal"
//                      (threshold|birefnet), "uv" (xatlas = default, unique
//                      chart space; box = faster projection), "band" (narrow-band
//                      DC remesh band width, default 1 — see --band), "model"
//                      (trellis|pixal3d — one directory holds both), "faces"
//                      (QEM decimation face target, >= 1000 or "auto" — see
//                      --faces), "atlas" (UV atlas px, 256..8192 or "auto"),
//                      "webp" (on|off; off = PNG textures), and for
//                      pixal3d the camera: "fov" (degrees), "mesh_scale",
//                      "extend_pixel", "texture" (on|off; off = geometry
//                      only, like --no-texture), and "normal_map" (on|off;
//                      on = bake a tangent-space normal map, like --normal-map).
//                      Returns model/gltf-binary.
//
//                    Multi-view (docs/spec/30-multiview.md): send several "image" parts
//                    of the same object plus "azimuths" (comma list, one per image in
//                    the same order; degrees or front|back|left|right|top|bottom) and
//                    optionally "elevations" (default 0). "azimuths" is required with
//                    more than one image. Also "blend_temp", "front_axis"
//                    (-y|+y|+x|-x|ref-z|ref-x) and "yaw_offset". One image and no angle
//                    fields is exactly the single-image request.
//   POST /refine      multipart/form-data with a "mesh" file part (GLB or OBJ, detected by
//                    magic) and one or more "image" parts plus every /generate field
//                    (angles, blend, seed, resolution, texture, faces, atlas, ...), and
//                    "refine_coords" (auto|encoder|voxel) and "restore_transform" (on|off).
//                    Re-generates the mesh from the view(s) on the mesh's own voxels
//                    (docs/spec/31-mesh-refine.md). model=pixal3d -> 400.
//   POST /jobs        asynchronous form of the two above (same fields; "task"=refine selects
//                    /refine) -> 202 {"id","status","position"}. Use it for long runs: the
//                    synchronous endpoints hold one request open for the whole run, which a
//                    browser or proxy may time out (Firefox gives up after 300 s).
//   GET  /jobs/{id}         -> {"status": queued|running|done|error, "position", "elapsed",
//                              "error"}
//   GET  /jobs/{id}/result  -> model/gltf-binary (409 while queued/running)
//   DELETE /jobs/{id}       cancels a queued job or frees a finished one (409 while running).
//                    Finished jobs are kept for 60 minutes.
//
// Launch-time defaults come from CLI flags (see trellis::parse_args);
// each request copies those defaults and applies its own overrides. The model
// directory is resolved once; each request runs the full pipeline via trellis_run()
// (per-stage load/free, like trellis-cli) on a single worker thread, in arrival order.
// Keeping the process resident avoids re-initializing the GPU backend on every request.
#include "trellis_args.h"
#include "trellis_run.h"
#include "httplib.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <map>
#include <memory>
#include <random>
#include <thread>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <utility>
#include <string>
#include <stdexcept>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

std::string read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

bool write_file_bytes(const std::string& path, const std::string& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(data.data(), (std::streamsize) data.size());
    return f.good();
}

// std::tmpnam on MSVC yields drive-root paths ("\sXXX.N") that a non-elevated
// process cannot write; stage scratch files in the real temp directory instead.
std::string temp_stem(const std::string& workdir = {}) {
    static std::atomic<unsigned> counter{0};
    std::error_code ec;
    std::filesystem::path dir = workdir.empty() ? std::filesystem::temp_directory_path(ec)
                                                : std::filesystem::absolute(workdir);
    if (ec) dir = ".";
    std::filesystem::create_directories(dir,ec);
    if (ec) throw std::runtime_error("cannot create request workdir: " + ec.message());
    auto n = counter.fetch_add(1);
    return (dir / ("trellis-req-" + std::to_string(n))).string();
}

void bad_request(httplib::Response& res, const std::string& msg) {
    res.status = 400;
    res.set_content(msg + "\n", "text/plain");
}

// Field overrides shared by every endpoint that runs the pipeline. Returns false (with a 400
// already set on `res`) on an invalid value.
bool apply_common_fields(const httplib::Request& req, trellis::TrellisParams& p, httplib::Response& res) {
    if (req.has_file("seed")) p.seed = (uint32_t) atoi(req.get_file_value("seed").content.c_str());
    if (req.has_file("resolution")) p.set_res(atoi(req.get_file_value("resolution").content.c_str()));
    if (req.has_file("bg_removal")) p.birefnet = (req.get_file_value("bg_removal").content == "birefnet") ? 1 : 0;
    if (req.has_file("uv")) p.xatlas = (req.get_file_value("uv").content == "xatlas");
    if (req.has_file("band")) p.band = atoi(req.get_file_value("band").content.c_str());
    if (req.has_file("atlas")) {
        std::string error;
        if (!trellis::parse_atlas_arg(req.get_file_value("atlas").content.c_str(), p, error)) {
            bad_request(res, error);
            return false;
        }
    }
    if (req.has_file("faces")) {
        std::string error;
        if (!trellis::parse_faces_arg(req.get_file_value("faces").content.c_str(), p, error) ||
            !trellis::check_faces_compat(p, error)) {   // launch-time --decim + per-request faces
            bad_request(res, error);
            return false;
        }
    }
    // Both families share one model directory (the Pixal3D flows carry a pixal3d_ prefix),
    // so a single resident server can serve either -- the client just has to say which.
    if (req.has_file("model")) {
        const std::string& m = req.get_file_value("model").content;
        if      (m == "pixal3d") p.family = trellis::ModelFamily::Pixal3D;
        else if (m == "trellis") p.family = trellis::ModelFamily::Trellis;
        else { bad_request(res, "model must be trellis or pixal3d"); return false; }
    }
    // Camera, for pixal3d. Unlike every other knob these describe the IMAGE rather than the
    // run, so a launch-time default is close to useless on a server taking arbitrary uploads.
    for (const auto& field : {std::pair<const char*, const char*>{"fov", "fov"},
                              {"mesh_scale", "mesh-scale"}, {"extend_pixel", "extend-pixel"}}) {
        if (!req.has_file(field.first)) continue;
        std::string error;
        if (!trellis::parse_camera_arg(field.second, req.get_file_value(field.first).content.c_str(), p, error)) {
            bad_request(res, error);
            return false;
        }
    }
    if (req.has_file("texture")) {
        const std::string& t = req.get_file_value("texture").content;
        if      (t == "off" || t == "0" || t == "false") p.texture = false;
        else if (t == "on"  || t == "1" || t == "true")  p.texture = true;
        else { bad_request(res, "texture must be on or off"); return false; }
    }
    if (req.has_file("normal_map")) {
        const std::string& n = req.get_file_value("normal_map").content;
        if      (n == "off" || n == "0" || n == "false") p.normal_map = 0;
        else if (n == "on"  || n == "1" || n == "true")  p.normal_map = 1;
        else { bad_request(res, "normal_map must be on or off"); return false; }
    }
    if (req.has_file("webp")) {
        const std::string& w = req.get_file_value("webp").content;
        p.webp = (w == "off" || w == "0" || w == "false") ? 0
               : (w == "on"  || w == "1" || w == "true")  ? 1 : -1;
    }
    // Multi-view blend knobs (meaningful with several images).
    if (req.has_file("blend_temp")) {
        const std::string& v = req.get_file_value("blend_temp").content;
        char* end = nullptr;
        const float f = std::strtof(v.c_str(), &end);
        if (end == v.c_str() || *end || !std::isfinite(f) || f <= 0) { bad_request(res, "blend_temp must be a positive number"); return false; }
        p.blend_temp = f;
    }
    if (req.has_file("front_axis")) {
        const std::string& v = req.get_file_value("front_axis").content;
        trellis::FrontAxis fa;
        if (!trellis::parse_front_axis(v, fa)) { bad_request(res, "front_axis must be one of -y, +y, +x, -x, ref-z, ref-x"); return false; }
        p.front_axis = v;
    }
    if (req.has_file("yaw_offset")) {
        const std::string& v = req.get_file_value("yaw_offset").content;
        char* end = nullptr;
        const float f = std::strtof(v.c_str(), &end);
        if (end == v.c_str() || *end || !std::isfinite(f)) { bad_request(res, "yaw_offset must be a number of degrees"); return false; }
        p.yaw_offset = f;
    }
    return true;
}

std::vector<std::string> split_commas(const std::string& s) {
    std::vector<std::string> out;
    size_t b = 0;
    for (;;) {
        const size_t e = s.find(',', b);
        std::string t = s.substr(b, e == std::string::npos ? std::string::npos : e - b);
        while (!t.empty() && std::isspace((unsigned char)t.front())) t.erase(t.begin());
        while (!t.empty() && std::isspace((unsigned char)t.back())) t.pop_back();
        out.push_back(t);
        if (e == std::string::npos) break;
        b = e + 1;
    }
    return out;
}

// The angles of `n` uploaded images from the "azimuths" / "elevations" fields. Each azimuth goes
// through parse_view_spec (so names work and the CLI's rules apply). Returns false with `error`.
bool parse_view_angles(const httplib::Request& req, size_t n, std::vector<trellis::ViewSpec>& out,
                       std::string& error) {
    out.assign(n, trellis::ViewSpec{});
    const bool has_az = req.has_file("azimuths"), has_el = req.has_file("elevations");
    if (n > 1 && !has_az) { error = "azimuths is required when more than one image is sent"; return false; }
    if (has_az) {
        const auto az = split_commas(req.get_file_value("azimuths").content);
        if (az.size() != n) {
            error = "azimuths has " + std::to_string(az.size()) + " entries for " + std::to_string(n) + " image(s)";
            return false;
        }
        for (size_t i = 0; i < n; ++i) {
            trellis::ViewSpec v;
            if (!trellis::parse_view_spec("x:" + az[i], v, error) || v.image != "x") {
                error = "bad azimuth '" + az[i] + "'";
                return false;
            }
            out[i].az_deg = v.az_deg;
            out[i].el_deg = v.el_deg;   // a name like "top" carries its elevation
        }
    }
    if (has_el) {
        const auto el = split_commas(req.get_file_value("elevations").content);
        if (el.size() != n) {
            error = "elevations has " + std::to_string(el.size()) + " entries for " + std::to_string(n) + " image(s)";
            return false;
        }
        for (size_t i = 0; i < n; ++i) {
            char* end = nullptr;
            const float f = std::strtof(el[i].c_str(), &end);
            if (el[i].empty() || *end || !std::isfinite(f) || f < -90 || f > 90) {
                error = "bad elevation '" + el[i] + "' (degrees within [-90, 90])";
                return false;
            }
            out[i].el_deg = f;
        }
    }
    return true;
}

std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) break;
                out += c;
        }
    }
    return out;
}

void json_error(httplib::Response& res, int status, const std::string& msg) {
    res.status = status;
    res.set_content("{\"error\":\"" + json_escape(msg) + "\"}", "application/json");
}

// One pipeline run: validated params plus the uploaded bytes, staged to disk only when the
// worker picks it up.
struct Job {
    enum class State { Queued, Running, Done, Error, Cancelled };
    std::string id;
    bool refine = false;
    bool sync = false;               // /generate, /refine: the handler waits and then drops it
    trellis::TrellisParams p;
    std::string stem;
    std::vector<std::string> staged; // file paths, parallel to `bytes`
    std::vector<std::string> bytes;
    size_t n_images = 0;
    State state = State::Queued;
    std::string glb;
    std::string error;
    std::chrono::steady_clock::time_point created = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point started, finished;
};

const char* state_name(Job::State s) {
    switch (s) {
        case Job::State::Queued:    return "queued";
        case Job::State::Running:   return "running";
        case Job::State::Done:      return "done";
        case Job::State::Error:     return "error";
        case Job::State::Cancelled: return "cancelled";
    }
    return "?";
}

// Validates a /generate, /refine or /jobs request into a Job. Returns nullptr with a 400 (or
// 500) already set on `res`.
std::shared_ptr<Job> make_job(const httplib::Request& req, httplib::Response& res,
                              const trellis::TrellisParams& base, bool refine) {
    if (refine && !req.has_file("mesh")) {
        json_error(res, 400, "missing 'mesh' file part (GLB or OBJ)");
        return nullptr;
    }
    if (!req.has_file("image")) {
        json_error(res, 400, "missing 'image' file part");
        return nullptr;
    }
    const auto images = req.get_file_values("image");

    auto job = std::make_shared<Job>();
    job->refine = refine;
    job->n_images = images.size();
    // Per-request params start from the launch defaults, then apply overrides.
    trellis::TrellisParams& p = job->p;
    p = base;
    if (!apply_common_fields(req, p, res)) return nullptr;
    std::vector<trellis::ViewSpec> angles;
    {
        std::string error;
        if (!parse_view_angles(req, images.size(), angles, error)) { bad_request(res, error); return nullptr; }
        if ((int)images.size() > trellis::kMaxViews) {
            bad_request(res, "at most " + std::to_string(trellis::kMaxViews) + " images are supported");
            return nullptr;
        }
        if ((images.size() > 1 || refine) && p.family == trellis::ModelFamily::Pixal3D) {
            bad_request(res, "--model pixal3d does not support multi-view conditioning (--view) or --refine-mesh yet;"
                             " use --model trellis, or pass a single image.");
            return nullptr;
        }
    }

    std::string mesh_bytes;
    if (refine) {
        mesh_bytes = req.get_file_value("mesh").content;
        if (mesh_bytes.size() < 4) { bad_request(res, "mesh part is empty"); return nullptr; }
        if (req.has_file("refine_coords")) {
            const std::string& v = req.get_file_value("refine_coords").content;
            if      (v == "auto")    p.refine_coords = -1;
            else if (v == "voxel")   p.refine_coords = 0;
            else if (v == "encoder") p.refine_coords = 1;
            else { bad_request(res, "refine_coords must be auto, encoder or voxel"); return nullptr; }
        }
        if (req.has_file("restore_transform")) {
            const std::string& v = req.get_file_value("restore_transform").content;
            if      (v == "on" || v == "1" || v == "true")   p.refine_restore = true;
            else if (v == "off" || v == "0" || v == "false") p.refine_restore = false;
            else { bad_request(res, "restore_transform must be on or off"); return nullptr; }
        }
    }

    try {
        job->stem = temp_stem(p.retopo ? p.retopo_workdir : std::string{});
    } catch (const std::exception&) {
        json_error(res, 500, "failed to stage request files");
        return nullptr;
    }
    const std::string& stem = job->stem;
    // One image and no angle fields: exactly the historical request (p.image, no views).
    // Otherwise every image is a view: <stem>_v<i>.png at its azimuth/elevation.
    const bool multi = images.size() > 1 || req.has_file("azimuths") || req.has_file("elevations");
    if (multi) {
        p.image.clear();
        p.views.clear();
        for (size_t i = 0; i < images.size(); ++i) {
            job->staged.push_back(stem + "_v" + std::to_string(i) + ".png");
            job->bytes.push_back(images[i].content);
            p.views.push_back(trellis::ViewSpec{ job->staged.back(), angles[i].az_deg, angles[i].el_deg });
        }
    } else {
        p.image = stem + ".png";
        job->staged.push_back(p.image);
        job->bytes.push_back(images[0].content);
    }
    p.output = stem + ".glb";
    if (refine) {
        // Detected by magic: binary glTF, else treated as OBJ text (load_mesh re-checks).
        p.refine_mesh = stem + (mesh_bytes.compare(0, 4, "glTF") == 0 ? "_mesh.glb" : "_mesh.obj");
        job->staged.push_back(p.refine_mesh);
        job->bytes.push_back(std::move(mesh_bytes));
    }
    return job;
}

// Runs one job (worker thread only): stage inputs, trellis_run, read the GLB, clean up.
void execute(Job& job) {
    trellis::TrellisParams& p = job.p;
    for (size_t i = 0; i < job.staged.size(); ++i) {
        if (!write_file_bytes(job.staged[i], job.bytes[i])) {
            for (const auto& f : job.staged) std::remove(f.c_str());
            job.error = "failed to stage input image";
            return;
        }
    }
    fprintf(stderr, "[trellis-server] %s %s: %zu image(s), %zu-byte first image, seed %u, res %s, bg %s, uv %s, faces %d, atlas %s, webp %s, texture %s\n",
            job.refine ? "refine" : "generate", job.id.c_str(), job.n_images, job.bytes[0].size(), p.seed,
            p.cascade ? std::to_string(p.hr_res).c_str() : "512",
            p.birefnet < 0 ? "auto" : (p.birefnet ? "birefnet" : "threshold"), p.xatlas ? "xatlas" : "box",
            trellis::qem_face_target(p), p.tex > 0 ? std::to_string(p.tex).c_str() : "auto",
            p.webp < 0 ? "auto" : (p.webp ? "on" : "off"), p.texture ? "on" : "off");
    job.bytes.clear();
    job.bytes.shrink_to_fit();
    std::string error_message = "3D reconstruction failed";
    try {
        int rc = trellis_run(p);
        if (rc == 0) job.glb = read_file_bytes(p.output);
    } catch (const std::exception& e) {
        fprintf(stderr, "[trellis-server] generate failed: %s\n", e.what());
        error_message = e.what();
    }
    for (const auto& f : job.staged) std::remove(f.c_str());
    std::remove(p.output.c_str());
    // trellis_run also writes sibling debug artifacts; clean them up too.
    std::remove((job.stem + ".ply").c_str());
    std::remove((job.stem + "_base.png").c_str());
    if (job.glb.empty()) job.error = error_message;
}

// FIFO queue with a single worker: one pipeline run at a time, in arrival order. Finished
// /jobs entries are kept for kRetention so a client can fetch the result after a reload or a
// dropped connection; DELETE frees one early.
class JobQueue {
public:
    static constexpr std::chrono::minutes kRetention{60};

    JobQueue() : worker_([this] { run(); }) {}
    ~JobQueue() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        worker_.join();
    }

    void submit(const std::shared_ptr<Job>& job) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            prune_locked();
            job->id = new_id_locked();
            jobs_[job->id] = job;
            pending_.push_back(job);
        }
        cv_.notify_all();
    }

    // Blocks until the job leaves the queue/running states (the synchronous endpoints).
    void wait(const std::shared_ptr<Job>& job) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return job->state != Job::State::Queued && job->state != Job::State::Running; });
        jobs_.erase(job->id);
    }

    std::shared_ptr<Job> find(const std::string& id) {
        std::lock_guard<std::mutex> lk(mu_);
        prune_locked();
        auto it = jobs_.find(id);
        return it == jobs_.end() ? nullptr : it->second;
    }

    // Status JSON for GET /jobs/{id}.
    std::string status_json(const Job& job) {
        std::lock_guard<std::mutex> lk(mu_);
        size_t position = 0;
        for (size_t i = 0; i < pending_.size(); ++i)
            if (pending_[i].get() == &job) position = i + 1;
        const auto now = std::chrono::steady_clock::now();
        auto secs = [](auto d) { return std::chrono::duration<double>(d).count(); };
        std::string s = "{\"id\":\"" + job.id + "\",\"status\":\"" + state_name(job.state) + "\"";
        if (job.state == Job::State::Queued) s += ",\"position\":" + std::to_string(position);
        char buf[64];
        if (job.state == Job::State::Running) {
            snprintf(buf, sizeof buf, ",\"elapsed\":%.1f", secs(now - job.started));
            s += buf;
        }
        if (job.state == Job::State::Done || job.state == Job::State::Error) {
            snprintf(buf, sizeof buf, ",\"elapsed\":%.1f", secs(job.finished - job.started));
            s += buf;
        }
        if (job.state == Job::State::Done) s += ",\"size\":" + std::to_string(job.glb.size());
        if (job.state == Job::State::Error) s += ",\"error\":\"" + json_escape(job.error) + "\"";
        return s + "}";
    }

    // DELETE /jobs/{id}: a queued job is cancelled, a finished one is freed; a running job can't
    // be interrupted (trellis_run has no cancel hook). Returns the HTTP status.
    int remove(const std::string& id, std::string& msg) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = jobs_.find(id);
        if (it == jobs_.end()) { msg = "unknown job"; return 404; }
        Job& job = *it->second;
        if (job.state == Job::State::Running) { msg = "job is running and can't be cancelled"; return 409; }
        if (job.state == Job::State::Queued) {
            job.state = Job::State::Cancelled;
            for (auto p = pending_.begin(); p != pending_.end(); ++p)
                if (p->get() == &job) { pending_.erase(p); break; }
        }
        jobs_.erase(it);
        cv_.notify_all();
        return 200;
    }

    // Guards Job::state reads that pair with other fields.
    std::mutex& mutex() { return mu_; }

private:
    void run() {
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stop_ || !pending_.empty(); });
                if (stop_) return;
                job = pending_.front();
                pending_.pop_front();
                job->state = Job::State::Running;
                job->started = std::chrono::steady_clock::now();
            }
            execute(*job);
            {
                std::lock_guard<std::mutex> lk(mu_);
                job->state = job->error.empty() ? Job::State::Done : Job::State::Error;
                job->finished = std::chrono::steady_clock::now();
                prune_locked();
            }
            cv_.notify_all();
        }
    }

    void prune_locked() {
        const auto now = std::chrono::steady_clock::now();
        for (auto it = jobs_.begin(); it != jobs_.end();) {
            const Job& j = *it->second;
            const bool finished = j.state == Job::State::Done || j.state == Job::State::Error;
            if (!j.sync && finished && now - j.finished > kRetention) it = jobs_.erase(it);
            else ++it;
        }
    }

    std::string new_id_locked() {
        static std::mt19937_64 rng{std::random_device{}()};
        char buf[32];
        snprintf(buf, sizeof buf, "%u-%08x", ++counter_, (unsigned)(rng() & 0xffffffffu));
        return buf;
    }

    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Job>> pending_;
    std::map<std::string, std::shared_ptr<Job>> jobs_;
    unsigned counter_ = 0;
    bool stop_ = false;
    std::thread worker_;
};

}  // namespace

int main(int argc, char** argv) {
    // Stage progress goes to stdout, which is fully buffered when piped (e.g.
    // under Lemonade's output capture) — keep it line-visible for diagnostics.
    setvbuf(stdout, nullptr, _IONBF, 0);

    trellis::TrellisParams base;
    if (!trellis::parse_args(argc, argv, base)) {
        trellis::print_usage(argv[0], /*server=*/true);
        return base.help ? 0 : 1;
    }

    JobQueue queue;
    httplib::Server svr;

    // Trellis Studio (and any browser client) calls this server from a different
    // origin — a Tauri webview is tauri://localhost / http://tauri.localhost, and a
    // browser-served UI is another port — so every response needs permissive CORS
    // headers, and a multipart POST with non-simple headers may be preflighted with
    // OPTIONS. Applied to every route via the post-routing hook + a catch-all OPTIONS.
    svr.set_post_routing_handler([](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.set_header("Access-Control-Max-Age", "86400");
    });
    svr.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;  // headers added by the post-routing handler above
    });

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("ok", "text/plain");
    });

    // /generate and /refine: synchronous -- queue the job and hold the connection until it is
    // done. Kept for curl and older clients; a long run can outlast a browser's or proxy's
    // response timeout, which is what /jobs avoids.
    auto run_sync = [&](const httplib::Request& req, httplib::Response& res, bool refine) {
        auto job = make_job(req, res, base, refine);
        if (!job) return;
        job->sync = true;
        queue.submit(job);
        queue.wait(job);
        if (job->state == Job::State::Cancelled) { json_error(res, 500, "job was cancelled"); return; }
        if (job->state != Job::State::Done) { json_error(res, 500, job->error); return; }
        res.set_content(job->glb.data(), job->glb.size(), "model/gltf-binary");
    };
    svr.Post("/generate", [&](const httplib::Request& req, httplib::Response& res) { run_sync(req, res, false); });
    svr.Post("/refine",   [&](const httplib::Request& req, httplib::Response& res) { run_sync(req, res, true); });

    // Asynchronous jobs: POST /jobs takes the /generate form (plus "task"=refine with a "mesh"
    // part for /refine) and answers 202 {"id",...} at once; GET /jobs/{id} reports
    // queued (with position) / running / done / error; GET /jobs/{id}/result returns the GLB;
    // DELETE /jobs/{id} cancels a queued job or frees a finished one.
    svr.Post("/jobs", [&](const httplib::Request& req, httplib::Response& res) {
        bool refine = false;
        if (req.has_file("task")) {
            const std::string& t = req.get_file_value("task").content;
            if      (t == "refine")   refine = true;
            else if (t != "generate") { json_error(res, 400, "task must be generate or refine"); return; }
        }
        auto job = make_job(req, res, base, refine);
        if (!job) return;
        queue.submit(job);
        res.status = 202;
        res.set_content(queue.status_json(*job), "application/json");
    });
    svr.Get(R"(/jobs/([\w-]+))", [&](const httplib::Request& req, httplib::Response& res) {
        auto job = queue.find(req.matches[1]);
        if (!job) { json_error(res, 404, "unknown job (finished jobs are kept for 60 minutes)"); return; }
        res.set_content(queue.status_json(*job), "application/json");
    });
    svr.Get(R"(/jobs/([\w-]+)/result)", [&](const httplib::Request& req, httplib::Response& res) {
        auto job = queue.find(req.matches[1]);
        if (!job) { json_error(res, 404, "unknown job (finished jobs are kept for 60 minutes)"); return; }
        Job::State st;
        {
            std::lock_guard<std::mutex> lk(queue.mutex());
            st = job->state;
        }
        if (st == Job::State::Error) { json_error(res, 500, job->error); return; }
        if (st != Job::State::Done)  { json_error(res, 409, std::string("job is ") + state_name(st)); return; }
        res.set_content(job->glb.data(), job->glb.size(), "model/gltf-binary");
    });
    svr.Delete(R"(/jobs/([\w-]+))", [&](const httplib::Request& req, httplib::Response& res) {
        std::string msg;
        const int status = queue.remove(req.matches[1], msg);
        if (status != 200) { json_error(res, status, msg); return; }
        res.set_content("{\"ok\":true}", "application/json");
    });

    fprintf(stderr, "[trellis-server] models=%s gpu=%d listening on http://%s:%d\n",
            base.models.c_str(), base.gpu, base.host.c_str(), base.port);
    if (!svr.listen(base.host, base.port)) {
        fprintf(stderr, "[trellis-server] failed to bind %s:%d\n", base.host.c_str(), base.port);
        return 1;
    }
    return 0;
}
