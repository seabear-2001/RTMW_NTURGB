cat > ~/autodl_mihomo_x86.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# AutoDL / Ubuntu container / x86_64 专用 mihomo 代理脚本
# 用法：
#   bash ~/autodl_mihomo_x86.sh install
#   bash ~/autodl_mihomo_x86.sh start
#   bash ~/autodl_mihomo_x86.sh stop
#   bash ~/autodl_mihomo_x86.sh restart
#   bash ~/autodl_mihomo_x86.sh status
#   bash ~/autodl_mihomo_x86.sh test

MIHOMO_VERSION="${MIHOMO_VERSION:-1.19.27}"
GH_PROXY="${GH_PROXY:-https://gh-proxy.com/}"

MIHOMO_DIR="/etc/mihomo"
MIHOMO_BIN="/usr/local/bin/mihomo"
CONFIG_FILE="${MIHOMO_DIR}/config.yaml"
SUB_FILE="${MIHOMO_DIR}/sub_url"
LOG_FILE="/tmp/mihomo.log"
PID_FILE="/tmp/mihomo.pid"

PROXY_HTTP="http://127.0.0.1:7890"
PROXY_SOCKS="socks5h://127.0.0.1:7890"

require_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "请在 root 用户下运行，AutoDL 容器默认就是 root。"
    exit 1
  fi
}

check_arch() {
  ARCH="$(uname -m)"
  if [[ "$ARCH" != "x86_64" && "$ARCH" != "amd64" ]]; then
    echo "当前架构是: $ARCH"
    echo "本脚本只适用于 x86_64 / amd64。"
    exit 1
  fi
}

clear_bad_proxy() {
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true
  git config --global --unset http.proxy 2>/dev/null || true
  git config --global --unset https.proxy 2>/dev/null || true
}

install_deps() {
  echo "[1/7] 安装依赖..."
  apt update
  apt install -y curl ca-certificates gzip python3-yaml net-tools procps
}

install_mihomo() {
  echo "[2/7] 下载 mihomo x86_64 版本..."

  mkdir -p "$MIHOMO_DIR"
  chmod 700 "$MIHOMO_DIR"

  URL="${GH_PROXY}https://github.com/MetaCubeX/mihomo/releases/download/v${MIHOMO_VERSION}/mihomo-linux-amd64-v1-v${MIHOMO_VERSION}.gz"

  echo "下载地址: $URL"

  curl --noproxy '*' -L --connect-timeout 20 --retry 5 \
    "$URL" \
    -o /tmp/mihomo.gz

  gzip -dc /tmp/mihomo.gz > "$MIHOMO_BIN"
  chmod +x "$MIHOMO_BIN"

  echo "mihomo 版本："
  "$MIHOMO_BIN" -v
}

download_mmdb() {
  echo "[3/7] 下载 Country.mmdb..."

  mkdir -p "$MIHOMO_DIR"

  curl --noproxy '*' -L --connect-timeout 20 --retry 5 \
    "${GH_PROXY}https://github.com/MetaCubeX/meta-rules-dat/releases/download/latest/country.mmdb" \
    -o "${MIHOMO_DIR}/Country.mmdb"

  cp "${MIHOMO_DIR}/Country.mmdb" "${MIHOMO_DIR}/country.mmdb"

  ls -lh "${MIHOMO_DIR}"/*ountry*.mmdb
}

save_sub_url() {
  echo "[4/7] 配置订阅链接..."

  if [[ -n "${SUB_URL:-}" ]]; then
    echo "$SUB_URL" > "$SUB_FILE"
  elif [[ -f "$SUB_FILE" ]]; then
    echo "检测到已有订阅链接，继续使用: $SUB_FILE"
  else
    read -r -p "请粘贴你的机场 Clash/Mihomo 订阅链接: " INPUT_SUB_URL
    echo "$INPUT_SUB_URL" > "$SUB_FILE"
  fi

  chmod 600 "$SUB_FILE"
}

update_config() {
  echo "[5/7] 拉取并修复订阅配置..."

  SUB_URL_VALUE="$(cat "$SUB_FILE")"

  curl --noproxy '*' -fL \
    -H "User-Agent: clash.meta" \
    -H "Accept: text/yaml,text/plain,*/*" \
    "$SUB_URL_VALUE" \
    -o /tmp/mihomo_sub.yaml

  python3 - <<'PY'
import yaml
from pathlib import Path

src = Path("/tmp/mihomo_sub.yaml")
dst = Path("/etc/mihomo/config.yaml")

text = src.read_text(encoding="utf-8", errors="ignore")
data = yaml.safe_load(text)

if not isinstance(data, dict):
    print("订阅内容不是 Clash/Mihomo YAML 配置，前 500 字如下：")
    print(text[:500])
    raise SystemExit(1)

# 本地代理端口
data["mixed-port"] = 7890
data["allow-lan"] = False
data["bind-address"] = "127.0.0.1"
data["external-controller"] = "127.0.0.1:9090"
data["log-level"] = "info"

# AutoDL 容器通常没有 TUN 权限，必须关闭
if isinstance(data.get("tun"), dict):
    data["tun"]["enable"] = False
else:
    data["tun"] = {"enable": False}

# 避免某些配置强依赖自动下载 geodata
data["geodata-mode"] = False

# DNS 简单保守配置
dns = data.get("dns")
if not isinstance(dns, dict):
    dns = {}

dns["enable"] = True
dns["listen"] = "127.0.0.1:1053"
dns.setdefault("enhanced-mode", "fake-ip")
dns.setdefault("fake-ip-range", "198.18.0.1/16")
dns.setdefault("nameserver", ["223.5.5.5", "119.29.29.29", "8.8.8.8"])
dns.setdefault("fallback", ["https://1.1.1.1/dns-query", "https://8.8.8.8/dns-query"])
data["dns"] = dns

dst.write_text(
    yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
    encoding="utf-8"
)
PY

  chmod 600 "$CONFIG_FILE"
  echo "配置文件已生成: $CONFIG_FILE"
}

stop_mihomo() {
  echo "停止 mihomo..."

  if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(cat "$PID_FILE" || true)"
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
      kill "$OLD_PID" 2>/dev/null || true
      sleep 1
    fi
    rm -f "$PID_FILE"
  fi

  pkill mihomo 2>/dev/null || true
}

start_mihomo() {
  echo "[6/7] 启动 mihomo..."

  stop_mihomo >/dev/null 2>&1 || true

  nohup "$MIHOMO_BIN" -d "$MIHOMO_DIR" -f "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"

  sleep 5

  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "mihomo 启动失败，日志如下："
    tail -n 120 "$LOG_FILE" || true
    exit 1
  fi

  echo "mihomo 已后台启动，PID: $(cat "$PID_FILE")"
}

write_proxy_env() {
  echo "[7/7] 写入代理环境变量..."

  cat > /etc/profile.d/mihomo_proxy.sh <<EOS
export http_proxy=${PROXY_HTTP}
export https_proxy=${PROXY_HTTP}
export HTTP_PROXY=${PROXY_HTTP}
export HTTPS_PROXY=${PROXY_HTTP}
export all_proxy=${PROXY_SOCKS}
export ALL_PROXY=${PROXY_SOCKS}
EOS

  chmod +x /etc/profile.d/mihomo_proxy.sh

  if ! grep -q "mihomo_proxy.sh" ~/.bashrc 2>/dev/null; then
    echo 'source /etc/profile.d/mihomo_proxy.sh' >> ~/.bashrc
  fi

  # 给当前脚本子进程使用；父 shell 需要用户手动 source
  export http_proxy="$PROXY_HTTP"
  export https_proxy="$PROXY_HTTP"
  export HTTP_PROXY="$PROXY_HTTP"
  export HTTPS_PROXY="$PROXY_HTTP"
  export all_proxy="$PROXY_SOCKS"
  export ALL_PROXY="$PROXY_SOCKS"

  echo "代理环境变量已写入 /etc/profile.d/mihomo_proxy.sh 和 ~/.bashrc"
  echo "当前终端请执行："
  echo "source /etc/profile.d/mihomo_proxy.sh"
}

test_proxy() {
  echo "测试 127.0.0.1:7890..."

  if ! netstat -lntp 2>/dev/null | grep -q ':7890'; then
    echo "7890 端口未监听。mihomo 可能没有启动。"
    echo "日志："
    tail -n 120 "$LOG_FILE" 2>/dev/null || true
    exit 1
  fi

  echo
  echo "测试 Google generate_204："
  curl -x "$PROXY_HTTP" -I --connect-timeout 20 https://www.google.com/generate_204 || true

  echo
  echo "测试 GitHub："
  curl -x "$PROXY_HTTP" -I --connect-timeout 20 https://github.com || true
}

status_mihomo() {
  echo "进程："
  ps aux | grep '[m]ihomo' || echo "mihomo 未运行"

  echo
  echo "端口："
  netstat -lntp 2>/dev/null | grep -E '7890|9090|1053' || echo "未看到 7890/9090/1053 监听"

  echo
  echo "最近日志："
  tail -n 80 "$LOG_FILE" 2>/dev/null || echo "暂无日志"
}

install_all() {
  require_root
  check_arch
  clear_bad_proxy
  install_deps
  install_mihomo
  download_mmdb
  save_sub_url
  update_config
  start_mihomo
  write_proxy_env
  test_proxy

  echo
  echo "安装完成。"
  echo "当前终端启用代理：source /etc/profile.d/mihomo_proxy.sh"
  echo "测试代理：bash ~/autodl_mihomo_x86.sh test"
  echo "查看状态：bash ~/autodl_mihomo_x86.sh status"
  echo "停止代理：bash ~/autodl_mihomo_x86.sh stop"
}

case "${1:-install}" in
  install)
    install_all
    ;;
  start)
    require_root
    clear_bad_proxy
    if [[ ! -f "$CONFIG_FILE" ]]; then
      echo "找不到配置文件，请先运行：bash ~/autodl_mihomo_x86.sh install"
      exit 1
    fi
    start_mihomo
    write_proxy_env
    test_proxy
    ;;
  stop)
    require_root
    stop_mihomo
    echo "已停止 mihomo。"
    ;;
  restart)
    require_root
    clear_bad_proxy
    stop_mihomo
    start_mihomo
    write_proxy_env
    test_proxy
    ;;
  update)
    require_root
    clear_bad_proxy
    save_sub_url
    update_config
    stop_mihomo
    start_mihomo
    test_proxy
    ;;
  status)
    status_mihomo
    ;;
  test)
    test_proxy
    ;;
  logs)
    tail -f "$LOG_FILE"
    ;;
  *)
    echo "未知命令: ${1:-}"
    echo "可用命令: install | start | stop | restart | update | status | test | logs"
    exit 1
    ;;
esac
EOF

chmod +x ~/autodl_mihomo_x86.sh