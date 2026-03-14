#ifndef UTILS_IOCOLOR_H
#define UTILS_IOCOLOR_H

// Color macros for formatted output
#define RED(MSG) "\033[31m" << MSG << "\033[0m"
#define BOLDRED(MSG) "\033[1m\033[31m" << MSG << "\033[0m"

#define GREEN(MSG) "\033[32m" << MSG << "\033[0m"
#define BOLDGREEN(MSG) "\033[1m\033[32m" << MSG << "\033[0m"

#define YELLOW(MSG) "\033[33m" << MSG << "\033[0m"
#define BOLDYELLOW(MSG) "\033[1m\033[33m" << MSG << "\033[0m"

#define BLUE(MSG) "\033[34m" << MSG << "\033[0m"
#define BOLDBLUE(MSG) "\033[1m\033[34m" << MSG << "\033[0m"

#define MAGENTA(MSG) "\033[35m" << MSG << "\033[0m"
#define BOLDMAGENTA(MSG) "\033[1m\033[35m" << MSG << "\033[0m"

#define CYAN(MSG) "\033[36m" << MSG << "\033[0m"
#define BOLDCYAN(MSG) "\033[1m\033[36m" << MSG << "\033[0m"

namespace IOColor {
static constexpr const char* RESET = "\033[0m";

static constexpr const char* BOLD = "\033[1m";
static constexpr const char* ITALIC = "\033[3m";

static constexpr const char* DEFAULT_FG = "\033[30m";
static constexpr const char* RED_FG = "\033[31m";
static constexpr const char* GREEN_FG = "\033[32m";
static constexpr const char* YELLOW_FG = "\033[33m";
static constexpr const char* BLUE_FG = "\033[34m";
static constexpr const char* MAGENTA_FG = "\033[35m";
static constexpr const char* CYAN_FG = "\033[36m";
} // namespace IOColor

#endif // UTILS_IOCOLOR_Hz