#!/bin/bash

# 安装 Python 3.11.12 的脚本（适用于无sudo权限的RHEL/CentOS）
# 将安装到用户目录下并更新用户级软链接

PYTHON_VERSION="3.11.12"
PYTHON_SOURCE_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"
INSTALL_DIR="${HOME}/.local/python-${PYTHON_VERSION}"
DEPENDENCIES="wget gcc zlib-devel bzip2 bzip2-devel readline-devel sqlite sqlite-devel openssl-devel libffi-devel"

# 1. 修复系统证书问题（临时方案）
echo "修复系统证书问题..."
mkdir -p /etc/pki/tls/certs
curl -sS https://curl.se/ca/cacert.pem -o /etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt

# 2. 安装依赖（使用root权限直接调用yum）
echo "安装编译依赖..."
yum install -y $DEPENDENCIES || {
    echo "依赖安装失败，尝试继续编译..."
}

# 3. 下载并编译Python
echo "下载 Python ${PYTHON_VERSION}..."
cd /tmp
wget --no-check-certificate $PYTHON_SOURCE_URL
tar -xf Python-${PYTHON_VERSION}.tgz
cd Python-${PYTHON_VERSION}

echo "编译安装到 ${INSTALL_DIR}..."
./configure --enable-optimizations --prefix=$INSTALL_DIR
make -j $(nproc)
make install

# 4. 创建用户级软链接
echo "创建用户级软链接..."
mkdir -p ${HOME}/.local/bin
ln -sf "${INSTALL_DIR}/bin/python3.11" "${HOME}/.local/bin/python"
ln -sf "${INSTALL_DIR}/bin/pip3.11" "${HOME}/.local/bin/pip"

# 5. 更新PATH
echo "export PATH=\${HOME}/.local/bin:\$PATH" >> ${HOME}/.bashrc
source ${HOME}/.bashrc

# 验证
echo -e "\n验证安装："
${HOME}/.local/bin/python --version
${HOME}/.local/bin/pip --version

echo -e "\n安装完成！请重新登录或运行: source ~/.bashrc"
echo "然后使用命令: python 或 pip"