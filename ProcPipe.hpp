#ifndef __WIN32
#ifndef _PROCPIPE_HPP
#define _PROCPIPE_HPP
#include <vector>
#include <tuple>
#include <string>
#include <string_view>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <csignal>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/socket.h>



template<bool redir_stdin, bool redir_stdout, bool redir_stderr>
class ProcPipe {
    struct Pipe {
        int readFd = -1, writeFd = -1;

        auto make() {
            return pipe(reinterpret_cast<int*>(this));
        }
        ~Pipe() {
            close(readFd); close(writeFd);
        }
    };
    struct Redirect {
        Pipe *pipe;
        int fd;
        bool output;
        int fdbak = -1;
    };

    constexpr static int errExit = 48;
    pid_t pid = 0;
    Pipe stdin,
         stdout,
         stderr;

    template<unsigned size>
    auto recvFrom(int fd) {
        static_assert (size != 0, "Can't read zero bytes");
        std::vector<char> fres(size);
        ssize_t bytes_read;
        if ((bytes_read = read(fd, fres.data(), fres.size())) < 0) {
            throw FdError("Failed to read() from stdout");
        }
        fres.resize(bytes_read);
        return fres;
    }

public:
    struct ExecutionError : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };
    struct AlreadyRunning : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };
    struct FdError : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    ProcPipe() {}
    template<typename... Args>
    ProcPipe(Args&&... args) {
        start(args...);
    }
    ~ProcPipe() {
        terminate();
    }

    void send(std::string_view str) {
        static_assert (redir_stdin, "Can't write to stdin if not redirected");
        if (write(stdin.writeFd, str.data(), str.size()) < 0) {
            throw FdError("Failed to write() to stdin");
        }
    }

    template<unsigned size>
    auto recvStd() {
        static_assert (redir_stdout, "Can't read from stdout if not redirected");
        return recvFrom<size>(stdout.readFd);
    }

    template<unsigned size>
    auto recvErr() {
        static_assert (redir_stderr, "Can't read from stdout if not redirected");
        return recvFrom<size>(stderr.readFd);
    }

    auto makeRedirs() {
        constexpr int redirs_size = redir_stdin + redir_stdout + redir_stderr;
        std::array<Redirect, redirs_size> redirs = {};
        {
            int idx = 0;
            if constexpr(redir_stdin) {
                redirs[idx++] = {&stdin, STDIN_FILENO, true};
            }
            if constexpr(redir_stdout) {
                redirs[idx++] = {&stdout, STDOUT_FILENO, false};
            }
            if constexpr(redir_stderr) {
                redirs[idx++] = {&stderr, STDERR_FILENO, false};
            }
        }
        return redirs;
    }

    template<typename... Args>
    void start(Args&&... args) {
        if (pid) {
            throw AlreadyRunning("Tried to run process in an instance where it is already running");
        } else {
            // Make redirects
            auto redirs = makeRedirs();
            // Redirect fds
            for (auto& io : redirs) {
                // Backup fd
                io.fdbak = dup(io.fd);
                // Create new pipe
                io.pipe->make();
                dup2((io.output ? io.pipe->readFd : io.pipe->writeFd), io.fd);
            }
            // Run process
            pid = fork();
            if (pid == 0) {
                const auto executable = std::get<0>(std::tuple{args...});
                execlp(executable, args..., nullptr);
                perror((std::string("Failed to launch ")+executable).c_str());
                exit(errExit);
            }
            // Restore fds
            for (const auto& io : redirs) {
                // Restore
                dup2(io.fdbak, io.fd);
            }
        }
    }

    auto waitExit() noexcept {
        if (pid) {
            int status = 0;
            waitpid(pid, &status, 0);
            pid = 0;
            return status;
        } else {
            return -1;
        }
    }

    auto terminate() noexcept {
        if (pid) {
            ::kill(pid, SIGTERM);
            return waitExit();
        } else {
            return -1;
        }
    }
    auto kill() noexcept {
        ::kill(pid, SIGKILL);
    }

    auto isRunning() noexcept {
        return !(::kill(pid, 0) < 0);
    }
};
#endif
#endif
