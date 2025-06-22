#!/bin/bash

# 安装 Python 3.11.12 的脚本（适用于问题系统）
# 直接下载静态编译版本，无需本地编译

PYTHON_VERSION="3.11.12"
INSTALL_DIR="${HOME}/.local/python-${PYTHON_VERSION}"
TAR_FILE="python-${PYTHON_VERSION}-static.tar.xz"
DOWNLOAD_URL="https://github.com/indygreg/python-build-standalone/releases/download/20240107/cpython-${PYTHON_VERSION}+20240107-x86_64-unknown-linux-gnu-install_only.tar.gz"

# 1. 创建安装目录
mkdir -p ${INSTALL_DIR}

# 2. 下载预编译版本
echo "下载预编译Python ${PYTHON_VERSION}..."
cd /tmp
if ! wget --no-check-certificate ${DOWNLOAD_URL} -O ${TAR_FILE}; then
    echo "下载失败，请检查网络连接"
    exit 1
fi

# 3. 解压安装
echo "解压到 ${INSTALL_DIR}..."
tar -xzf ${TAR_FILE} -C ${INSTALL_DIR} --strip-components=1

# 4. 创建软链接
echo "创建用户级软链接..."
mkdir -p ${HOME}/.local/bin
ln -sf "${INSTALL_DIR}/bin/python3" "${HOME}/.local/bin/python"
ln -sf "${INSTALL_DIR}/bin/pip3" "${HOME}/.local/bin/pip"

# 5. 更新PATH
echo "export PATH=\${HOME}/.local/bin:\$PATH" >> ${HOME}/.bashrc
source ${HOME}/.bashrc

# 验证
echo -e "\n验证安装："
${HOME}/.local/bin/python --version
${HOME}/.local/bin/pip --version

echo -e "\n安装完成！请重新登录或运行: source ~/.bashrc"
echo "使用命令: python 或 pip"