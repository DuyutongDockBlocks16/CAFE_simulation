#!/bin/bash

# 安装 Python 3.11.12 并设置为默认版本的脚本
# 需要以 root 权限运行

PYTHON_VERSION="3.11.12"
PYTHON_SOURCE_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"
INSTALL_DIR="/usr/local/python${PYTHON_VERSION}"
DEPENDENCIES="wget build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev"

# 检查是否以root运行
if [ "$(id -u)" -ne 0 ]; then
    echo "请使用root权限运行此脚本！"
    echo "可以尝试: sudo bash $0"
    exit 1
fi

# 安装编译依赖
echo "安装编译依赖..."
apt-get update -y
apt-get install -y $DEPENDENCIES

# 下载Python源码
echo "下载 Python ${PYTHON_VERSION}..."
cd /tmp
wget $PYTHON_SOURCE_URL
tar -xf Python-${PYTHON_VERSION}.tgz
cd Python-${PYTHON_VERSION}

# 编译安装
echo "编译安装 Python ${PYTHON_VERSION}..."
./configure --enable-optimizations --prefix=$INSTALL_DIR
make -j $(nproc)
make altinstall

# 创建软链接
echo "创建软链接..."
ln -sf "${INSTALL_DIR}/bin/python3.11" /usr/local/bin/python3
ln -sf "${INSTALL_DIR}/bin/python3.11" /usr/local/bin/python
ln -sf "${INSTALL_DIR}/bin/pip3.11" /usr/local/bin/pip3
ln -sf "${INSTALL_DIR}/bin/pip3.11" /usr/local/bin/pip

# 验证安装
echo -e "\n安装完成，验证版本："
python --version
pip --version

echo -e "\nPython ${PYTHON_VERSION} 已安装并设置为默认版本！"