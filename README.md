# Sakura模型部署教程

### 写在前面

transformers部署模型目前需要至少16G显存，如果你的显卡显存不足16G，可以尝试使用llama.cpp gguf量化模型在cpu上运行。

本教程分为以下几个部分：

- 安装环境
- 本地翻译轻小说Epub或文本文件
- 运行API后端并接入网站
- 可能遇到的问题

## 打包好的Windows环境包

考虑到复杂的Windows环境安装流程，这里提供一个在Windows上测试可用的环境包。

1. 下载压缩包 `sakura.tar.gz` 并解压缩到文件夹，链接如下，根据自己的网络环境选择其一下载即可：
    - [Google Drive](https://drive.google.com/file/d/1hxn8ECwvSF73crLTNeYRjNCSbEwqvMcs/view?usp=sharing)
    - [OneDrive(CN)](https://acgg-my.sharepoint.cn/:u:/g/personal/yukinon_srpr_cc/Ea4oE9XBvthAtyRfYn_JbuMB1CImDYOHcl77ryk_rXZIUw?e=rF8Jdx)，密码：sakura
2. 解压缩后，你现在应该有一个叫做 `sakura` 的文件夹，里面有一个 `python.exe` 文件。现在你可以用 Python/Anaconda 两种方式使用 Sakura 环境：
    - **Python**：直接使用里面的`python.exe`来运行 Python 脚本。注意使用的时候，不要用默认的 `python` 命令，而要使用绝对路径或相对路径指向下载的 `python.exe`。 例如：
        
        ```python
        python xxx.py				# 错误，使用的是系统的python解释器
        sakura/python.exe xxx.py	# 正确，使用的是下载的Sakura的python解释器
        ```
        
    - **Anaconda**：将解压缩的文件夹 `sakura` 放到 Anaconda 的 `envs` 目录下，运行`conda activate sakura`来应用环境。
3. 完成上面步骤之后，请参考《本地翻译》或《运行API后端》章节，来使用 Sakura。

## AutoDL镜像

1. 部署服务器

如果你想租显卡跑模型，可以使用提供的AutoDL镜像。

在AutoDL租用显卡的页面下方，选择社区镜像，搜索提供的镜像名字（目前直接搜索Sakura-13B即可），点击选中，具体如下图所示：

![Untitled](assets/autodl_1.png)

成功部署后，通过ssh连接至服务器，首先执行以下指令：

```bash
source /etc/network_turbo
cd /root/Sakura-13B-Galgame
git pull
conda activate sakura
```

1. 运行
    1. 在这之后，如果想翻译文本或者Epub轻小说，请跳转至本地翻译的[完成后打开终端，执行以下命令（注意系统）：](https://www.notion.so/661492a047534d5c98e88ee56f2cbcfe?pvs=21)小节 。
    2. 如果想运行API后端接入网站或其他程序，请跳转至运行API后端（直接安装）的[运行程序](https://www.notion.so/682662d8caea463c8eb86265df090651?pvs=21) 小节。
        1. 命令中`--model_name_or_path`的值需要改成`/root/Sakura-13B-Galgame/Sakura-13B-LNovel-v0_8-4bit`
2. 根据上文成功运行API后端后，如果想继续接入轻小说机翻机器人并生成翻译，需要继续执行以下操作（下面两小节二选一）：
    - 使用AutoDL自带的公网映射服务映射API到公网
        
        必须使用以下命令启动API后端，~~其中<user>和<pass>需要替换成你想设置的账号和密码，以防止出现安全问题：~~
        
        ```bash
        python server.py --model_name_or_path /root/Sakura-13B-Galgame/Sakura-13B-LNovel-v0_8-4bit --use_gptq_model --model_version 0.8 --trust_remote_code --listen 0.0.0.0:6006 --no-auth
        ```
        
        再次确认服务器中API监听的地址为0.0.0.0:6006，保证程序持续运行，打开AutoDL控制台，点击服务器右侧的“自定义服务”，将打开的网址复制下来。然后在后面添加`/api/v1/generate`
        
        ![Untitled](assets/autodl_2.png)
        
    - 映射API到本地
        
        首先打开AutoDL控制台，找到你的服务器的ssh命令，如下图所示：
        
        ![Untitled](assets/autodl_3.png)
        
        此时复制出来后，会出现类似`ssh -p 11451 root@senpai.1919810.com`的命令，其中11451是端口，senpai.1919810.com是域名，记住这两个信息，将下面的命令补全，其中<port>换成你的端口，<domain>换成你的域名。
        如果你使用了教程的默认参数启动API后端，则可以直接执行。如果你想更改本地监听的端口，则需要将`5000:127.0.0.1:5000`改成`你想要监听的端口:127.0.0.1:5000`。如果你更改了API的监听地址，还需要将`5000:你的API监听地址`
        
        ```bash
        ssh -N -L 5000:127.0.0.1:5000 root@<domain> -p <port>
        ```
        
        保证这个终端不关闭，此时AutoDL服务器上的API后端链接就映射到本地了。你可以使用`http://127.0.0.1:5000/api/v1/generate` 这个url，参考[接入网站：轻小说机翻机器人](https://www.notion.so/3e9aefb734d94d569c561cf47baae3ce?pvs=21) 这一章节的方式接入网站。
        

## 安装环境（Windows）

考虑到大部分用户使用的是Windows系统，教程会主要以Windows系统的视角提供部署教程。

1. 首先，你需要安装python3.10或以上版本（推荐python3.10）
2. 安装好python后，需要安装运行所必须的库。
    
    (1). torch. 首先，打开终端，输入指令`nvcc -V` ，并查看此处release后面显示的版本。
    
    ![Untitled](assets/cuda_1.png)
    
    如果你的CUDA版本是12.0以上，则运行以下命令：
    
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
    
    如果你的CUDA版本是11.x，则运行以下命令：
    
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    
    (2). bitsandbytes和auto-gptq. 
    
    - bitsandbytes：
    
    ```bash
    pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.40.2-py3-none-win_amd64.whl
    ```
    
    - auto-gptq：根据你的CUDA版本和Python版本进行安装，这里使用Python3.10作为例子：
    
    ```bash
    # CUDA 11.7（11.7及以前）
    pip install https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.4.2/auto_gptq-0.4.2+cu117-cp310-cp310-win_amd64.whl
    # CUDA 11.8（11.8及以后）
    pip install https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.4.2/auto_gptq-0.4.2+cu118-cp310-cp310-win_amd64.whl
    ```
    
    如果出现CUDA版本问题，可先尝试将CudaToolkit更新/降级至11.7/11.8版本
    
    (3). 其他依赖
    
    运行以下命令：
    
    ```bash
    pip3 install transformers==4.33.2 sentencepiece xformers
    ```
    

## 安装环境（Linux）

~~都会用Linux了，应该看了上面的Windows版教程就能装好了吧~~

## 本地翻译

1. 如果你想直接本地翻译轻小说Epub或文本：
    
    执行以下命令克隆项目仓库
    
    ```bash
    git clone https://github.com/SakuraLLM/Sakura-13B-Galgame.git
    ```
    
    克隆后进入该目录，将你想翻译的Epub文件或文本文件放入仓库根目录中，并重命名为novel.epub（或novel.txt），如果你想翻译Epub文件，还需要创建一个文件夹output。
    
    ![Untitled](assets/translate_1.png)
    
    完成后打开终端，执行以下命令（注意系统）：
    
    - 如果你想翻译Epub文件：
        
        ```bash
        # 参数说明：
        # --model_name_or_path：模型本地路径或者huggingface仓库id。
        # --model_version：模型版本，本仓库README表格中即可查看。可选范围：['0.1', '0.4', '0.5', '0.7', '0.8']
        # --use_gptq_model：如果模型为gptq量化模型，则需加此项；如是全量模型，则不需要添加。
        # --text_length：文本分块的最大单块文字数量。
        # --data_path：日文原文Epub小说文件路径。
        # --data_folder：批量翻译Epub小说时，小说所在的文件夹路径
        # --output_folder：翻译后的Epub文件输出路径（注意是文件夹路径）。
        # --trust_remote_code：是否允许执行外部命令（对于0.5，0.7，0.8版本模型需要加上这个参数，否则报错）。
        # --llama：如果你使用的模型是llama家族的模型（对于0.1，0.4版本），则需要加入此命令。
        # 以下为一个例子
        # Linux的例子
        python translate_epub.py \
            --model_name_or_path SakuraLLM/Sakura-13B-LNovel-v0_8-4bit \
            --trust_remote_code \
            --model_version 0.8 \
            --use_gptq_model \
            --text_length 512 \
            --data_path novel.epub \
            --output_folder output
        # Windows的例子
        python translate_epub.py --model_name_or_path SakuraLLM/Sakura-13B-LNovel-v0_8-4bit --trust_remote_code --model_version 0.8 --use_gptq_model --text_length 512 --data_path novel.epub --output_folder output
        ```
        
    - 如果你想翻译文本文件：
        
        ```bash
        # 参数说明：
        # --model_name_or_path：模型本地路径或者huggingface仓库id。
        # --model_version：模型版本，本仓库README表格中即可查看。可选范围：['0.1', '0.4', '0.5', '0.7', '0.8']
        # --use_gptq_model：如果模型为gptq量化模型，则需加此项；如是全量模型，则不需要添加。
        # --text_length：文本分块的最大单块文字数量。每块文字量将在text_length/2至text_length内随机选择。
        # --compare_text：是否需要输出中日对照文本，如需要，则需加此项；如不需要则不要添加。
        # --data_path：日文原文文件路径
        # --output_path：翻译(或对照)文本输出文件路径
        # --trust_remote_code：是否允许执行外部命令（对于0.5，0.7，0.8版本模型需要加上这个参数，否则报错。
        # --llama：如果你使用的模型是llama家族的模型（对于0.1，0.4版本），则需要加入此命令。
        # 以下为一个例子
        # Linux的例子
        python translate_novel.py \
            --model_name_or_path SakuraLLM/Sakura-13B-LNovel-v0_8-4bit \
            --trust_remote_code \
            --model_version 0.8 \
            --use_gptq_model \
            --text_length 512 \
            --data_path novel.txt \
            --output_path novel_translated.txt
        # Windows的例子
        python translate_novel.py --model_name_or_path SakuraLLM/Sakura-13B-LNovel-v0_8-4bit --trust_remote_code --model_version 0.8 --use_gptq_model --text_length 512 --data_path novel.txt --output_path novel_translated.txt
        ```
        
    
    这里，如果你位于中国大陆地区，那么到huggingface.co的网络可能存在无法连接的问题，导致无法拉取模型。你可以从modelscope仓库拉取模型并存到本地，并将上文命令里的`--model_name_or_path`的值改为本地模型文件夹的路径。
    

## 运行API后端（直接安装）

1. 克隆Github仓库，并进入仓库目录
    
    ```bash
    git clone https://github.com/SakuraLLM/Sakura-13B-Galgame
    ```
    
    克隆后，使用`git checkout dev_server`切换到api后端的分支。
    
2. 安装所需的环境（Windows）：先按照上文《安装环境（Windows）》安装完成后，执行下面的命令：
    
    ```jsx
    pip install scipy numpy fastapi[all] hypercorn coloredlogs dacite asyncio sse-starlette
    ```
    
3. 安装所需的环境（Linux）：可以直接执行下面的指令。需要上文中提到的cuda版本，如果是12.x则无需操作，如果是11.x请在requirements.txt里将第一行注释掉，并将第二行取消注释。
    
    ```bash
    pip install -r requirements.txt
    ```
    
4. 运行程序
    
    ```bash
    # 参数解释（与上文本地翻译相同的参数不再进行解释）：
    # --listen：指定要监听的IP和端口，格式为<IP>:<Port>，如127.0.0.1:5000。默认为127.0.0.1:5000
    # --auth：使用认证，访问API需要提供账户和密码。
    # --no-auth：不使用认证，如果将API暴露在公网可能会降低安全性。
    # --log：设置日志等级。
    # 下面为一个使用v0.8-4bit模型，同时不使用认证，监听127.0.0.1:5000的命令示例。
    # 这里模型默认从huggingface拉取，如果你已经将模型下载至本地，可以将--model_name_or_path参数的值指定为本地目录。
    python server.py --model_name_or_path SakuraLLM/Sakura-13B-LNovel-v0_8-4bit --use_gptq_model --model_version 0.8 --trust_remote_code --no-auth
    ```
    

## 运行API后端（Docker）

- 使用AutoDL开放的Docker镜像：[https://www.codewithgpu.com/i/SakuraLLM/Sakura-13B-Galgame/Sakura-13B-Galgame](https://www.codewithgpu.com/i/SakuraLLM/Sakura-13B-Galgame/Sakura-13B-Galgame)
- 使用Github仓库的Docker镜像

## 接入网站：轻小说机翻机器人

1. 如果你的API后端地址的IP不是127.0.0.1（或者域名不是localhost），可能会触发浏览器的
2. 将前面API后端的网址填入此处
    
    ![Untitled](assets/website_1.png)
    
    然后点击“测试Sakura”可以进行简单的测试。如果没有问题，则可以直接点击下方“更新Sakura”开始翻译。
    

## 可能遇到的问题

1. 使用llama.cpp后端运行API
    - 如果你是Windows系统，请通过以下命令安装新依赖：
    
    ```bash
    python -m pip install llama-cpp-python --prefer-binary --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117
    ```
    
    - 安装依赖后，使用以下例子启动API：
    
    ```jsx
    # --use_gpu：如果需要使用GPU推理则使用，只需要纯CPU推理的话请删掉。
    python server.py --listen 0.0.0.0:5000 --llama_cpp --trust_remote_code --model_name_or_path 你的GGUF模型位置 --model_version 0.8 --no-auth --log info --use_gpu
    ```
    
2. 使用llama.cpp后端运行脚本
    - 如果你是Windows系统，请通过以下命令安装新依赖：
    
    ```bash
    python -m pip install llama-cpp-python --prefer-binary --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117
    ```
    
    - 安装依赖后，请使用以下参数和例子（以翻译文本的脚本为例，翻译Epub的脚本同理）：
    
    ```bash
    # 参数说明：
    # --model_name_or_path：GGUF格式模型本地路径。
    # --model_version：模型版本，本仓库README表格中即可查看。可选范围：['0.1', '0.4', '0.5', '0.7', '0.8']
    # --text_length：文本分块的最大单块文字数量。
    # --compare_text：是否需要输出中日对照文本，如需要，则需加此项；如不需要则不要添加。
    # --data_path：日文原文文件路径
    # --output_path：翻译(或对照)文本输出文件路径
    # --trust_remote_code：是否允许执行外部命令（对于0.5，0.7，0.8版本模型需要加上这个参数，否则报错）
    # --llama_cpp：使用llama.cpp载入模型
    # --use_gpu：使用GPU载入llama.cpp支持的模型
    # --n_gpu_layers：设置载入GPU的模型层数。如果指定了use_gpu同时此处设置为0（或者不设置），则会将模型所有层都放入GPU中。如果没有指定use_gpu同时此处设置为0（或者不设置），则会进入纯CPU推理模式。
    # 以下为一个例子
    # Windows的例子
    python .\translate_novel.py --model_name_or_path 你的模型路径 --model_version 0.8 --data_path text.txt --output_path output.txt --llama_cpp --use_gpu --trust_remote_code
    ```
    
3. `ModuleNotFoundError: No module named ‘sampler_hijack’`
    
    少下了一个文件`sampler_hijack.py`，请完整克隆仓库。
    
4. 出现类似`SSLError/ConnectionError/MaxRetryError`等连接不上huggingface.co的报错
    
    中国大陆网络问题导致的，可重试或使用魔法；或者检查`model_name_or_path`参数有没有写错。
    
5. `CUDA extension not installed.`
    
    安装的autogptq与你的环境不匹配，需要卸载autogptq `pip uninstall autogptq` 之后，使用源码安装：
    
    ![Untitled](assets/autogptq_1.png)