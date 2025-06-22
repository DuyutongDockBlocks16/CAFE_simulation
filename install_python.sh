#!/bin/bash

# 安装 Python 3.11.12 的脚本（使用官方预编译版本）
PYTHON_VERSION="3.11.12"
INSTALL_DIR="${HOME}/.local/python-${PYTHON_VERSION}"
TAR_FILE="Python-${PYTHON_VERSION}.tgz"
DOWNLOAD_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/${TAR_FILE}"

# 1. 创建安装目录
mkdir -p ${INSTALL_DIR}

# 2. 下载官方源码包（带预编译选项）
echo "下载Python ${PYTHON_VERSION}..."
cd /tmp
if ! wget --no-check-certificate ${DOWNLOAD_URL}; then
    echo "下载失败，请尝试以下手动方案："
    echo "1. 在其他机器下载 ${DOWNLOAD_URL}"
    echo "2. 将文件上传到本机/tmp目录"
    echo "3. 重新运行此脚本"
    exit 1
fi

# 3. 解压并安装
echo "解压安装..."
tar -xzf ${TAR_FILE}
cd Python-${PYTHON_VERSION}

# 最小化编译（跳过优化以加快速度）
./configure --prefix=${INSTALL_DIR} --with-ensurepip=install
make -j$(nproc)
make install

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

echo -e "\n安装完成！请执行以下命令生效："
echo "source ~/.bashrc"