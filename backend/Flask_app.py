# from flask import Flask, request, jsonify
# import torch
# import pandas as pd
# import os
# from flask_caching import Cache
#
# from mega_rapid_api import fetch_mega_millions_data  # 引入你的 mega_rapid_api.py 文件中的函数
# from mega_statistic_api import fetch_mega_millions_statistic_data
#
# from power_rapid_api import fetch_power_ball_data
# from power_statistic_api import fetch_power_ball_statistic_data
# from gan_generator import Generator
# from flask_cors import CORS
#
# app = Flask(__name__)
# cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})
# CORS(app)  # 添加跨域支持
#
# # 配置参数
# latent_dim = 30  # 随机噪声维度
# num_classes = 70  # 主号码范围 (1-70)
# mega_classes = 25  # Mega Ball 范围 (1-25)
# number_dim = 5 * num_classes  # 主号码的总输出维度
# mega_dim = mega_classes  # Mega Ball 的输出维度
# condition_dim = number_dim + mega_dim  # 条件特征的总维度
#
# # **条件特征加载函数**
# def load_condition_features(file_path, num_classes=70, mega_classes=25):
#     """
#     从历史数据 CSV 文件中加载条件特征。
#
#     Args:
#         file_path (str): CSV 文件路径。
#         num_classes (int): 主号码范围。
#         mega_classes (int): Mega Ball 范围。
#
#     Returns:
#         torch.Tensor: 条件特征张量，大小为 (1, num_classes * 5 + mega_classes)。
#     """
#     # 读取 CSV 数据
#     data = pd.read_csv(file_path)
#
#     # 初始化条件特征张量
#     condition = torch.zeros(num_classes * 5 + mega_classes)
#
#     # 统计主号码分布
#     for i, col in enumerate(["Number1", "Number2", "Number3", "Number4", "Number5"]):
#         counts = data[col].value_counts(normalize=True, sort=False)  # 归一化概率
#         for number in range(1, num_classes + 1):
#             condition[i * num_classes + (number - 1)] = counts.get(number, 0)
#
#     # 统计 Mega Ball 分布
#     counts_mega = data["MegaBall"].value_counts(normalize=True, sort=False)
#     for number in range(1, mega_classes + 1):
#         condition[num_classes * 5 + (number - 1)] = counts_mega.get(number, 0)
#
#     # 返回条件特征张量，并添加批次维度
#     return condition.unsqueeze(0)
#
# # **加载条件特征**
# condition_features = load_condition_features("E:\\Pychram\\lotteryAI\\Lottery_data\\API_drawing_data.csv")
# print("条件特征加载完成，特征维度：", condition_features.size())
#
# # **加载生成器模型**
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_path = "E:\Pychram\lotteryAI\\backend\gan_generator.pth"
#
# G = Generator(latent_dim, condition_dim, number_dim, mega_dim).to(device)
#
# if os.path.exists(model_path):
#     G.load_state_dict(torch.load(model_path, map_location=device))
#     G.eval()  # 设置为评估模式
#     print("生成器模型加载成功！")
# else:
#     raise FileNotFoundError(f"未找到模型文件：{model_path}")
#
# # **推理函数**
# def generate_numbers(batch_size=1):
#     """
#     使用加载的生成器模型生成彩票号码。
#
#     Args:
#         batch_size (int): 生成号码的组数。
#
#     Returns:
#         tuple: 主号码和 Mega Ball 的生成结果。
#     """
#     def post_process_unique(numbers):
#         """
#         后处理：确保 Number1 到 Number5 唯一性。
#
#         Args:
#             numbers (torch.Tensor): 主号码张量。
#
#         Returns:
#             torch.Tensor: 经过后处理的主号码张量。
#         """
#         batch_size, _, num_classes = numbers.size()
#         result = torch.zeros_like(numbers)
#         for i in range(batch_size):
#             chosen = set()
#             for j in range(5):
#                 idx = torch.argmax(numbers[i, j]).item()
#                 while idx in chosen:
#                     numbers[i, j, idx] = 0  # 将重复的概率置 0
#                     idx = torch.argmax(numbers[i, j]).item()
#                 chosen.add(idx)
#                 result[i, j, idx] = 1
#         return result
#
#     with torch.no_grad():
#         # 生成随机噪声
#         noise = torch.randn(batch_size, latent_dim).to(device)
#         # 重复条件特征，匹配 batch_size
#         condition_sample = condition_features.repeat(batch_size, 1).to(device)
#         # 调用生成器
#         numbers, mega = G(noise, condition_sample)
#
#         # 后处理主号码
#         numbers = post_process_unique(numbers)
#         # 提取主号码和 Mega Ball
#         generated_numbers = torch.argmax(numbers, dim=-1).cpu().numpy() + 1
#         generated_mega = torch.argmax(mega, dim=-1).cpu().numpy() + 1
#
#         # 转换为 Python 的原生类型
#         generated_numbers = generated_numbers.tolist()
#         generated_mega = generated_mega.tolist()
#
#         return generated_numbers, generated_mega



from flask import Flask, request, jsonify
import torch
import pandas as pd
import os
from flask_caching import Cache
from gan_generator import Generator  # Mega Millions 的生成器类
from gan_power_generator import PowerBallGenerator # Power Ball 的生成器类
from flask_cors import CORS, cross_origin


from mega_rapid_api import fetch_mega_millions_data  # 引入你的 mega_rapid_api.py 文件中的函数
from mega_statistic_api import fetch_mega_millions_statistic_data

from power_rapid_api import fetch_power_ball_data
from power_statistic_api import fetch_power_ball_statistic_data




app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})
from flask_cors import CORS

# CORS(app, resources={r"/*": {"origins": "*"}})  # 允许所有前端访问
CORS(app, origins=["https://lotterygenai-frontend.onrender.com"])




# ---- 设置项目根路径 (统一路径用) ----
import os

# 当前文件的目录（即 backend/）
backend_dir = os.path.dirname(os.path.abspath(__file__))

# 项目根目录（即 backend/ 的上一级）
project_root = os.path.abspath(os.path.join(backend_dir, ".."))

# 定义一个辅助函数，快速获取任何相对到项目根路径的文件
def from_root(*relative_path_parts):
    return os.path.join(project_root, *relative_path_parts)




# Mega Millions 参数
mega_latent_dim = 30
mega_num_classes = 70
mega_mega_classes = 25
mega_condition_dim = mega_num_classes * 5 + mega_mega_classes

# Power Ball 参数
power_latent_dim = 30
power_num_classes = 69
power_mega_classes = 26
power_condition_dim = power_num_classes * 5 + power_mega_classes

# **加载 Mega Millions 条件特征**
def load_condition_features_mega(file_path, num_classes=70, mega_classes=25):
    """
    从历史数据 CSV 文件中加载条件特征。

    Args:
        file_path (str): CSV 文件路径。
        num_classes (int): 主号码范围。
        mega_classes (int): Mega Ball 范围。

    Returns:
        torch.Tensor: 条件特征张量，大小为 (1, num_classes * 5 + mega_classes)。
    """
    data = pd.read_csv(file_path)
    condition = torch.zeros(num_classes * 5 + mega_classes)
    for i, col in enumerate(["Number1", "Number2", "Number3", "Number4", "Number5"]):
        counts = data[col].value_counts(normalize=True, sort=False)
        for number in range(1, num_classes + 1):
            condition[i * num_classes + (number - 1)] = counts.get(number, 0)
    counts_mega = data["MegaBall"].value_counts(normalize=True, sort=False)
    for number in range(1, mega_classes + 1):
        condition[num_classes * 5 + (number - 1)] = counts_mega.get(number, 0)
    return condition.unsqueeze(0)

#mega_condition_features = load_condition_features_mega("Lottery_data/API_drawing_data.csv")      # Lottery_data/API_drawing_data.csv

csv_path = from_root("Lottery_data", "API_drawing_data.csv")
mega_condition_features = load_condition_features_mega(csv_path)


print("Mega Millions 条件特征加载完成，特征维度：", mega_condition_features.size())

# Power Ball 使用默认条件特征
power_condition_features = torch.zeros(power_condition_dim).unsqueeze(0)
print("Power Ball 使用默认条件特征，特征维度：", power_condition_features.size())

# **加载生成器模型**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mega Millions
#mega_model_path = "backend/gan_generator.pth"             # /Users/shuhaozhang/Documents/lotteryAI/backend/gan_generator.pth
mega_model_path = from_root("backend", "gan_generator.pth")

G_mega = Generator(mega_latent_dim, mega_condition_dim, mega_num_classes * 5, mega_mega_classes).to(device)

if os.path.exists(mega_model_path):
    G_mega.load_state_dict(torch.load(mega_model_path, map_location=device))
    G_mega.eval()
    print("Mega Millions 生成器模型加载成功！")
else:
    raise FileNotFoundError(f"未找到模型文件：{mega_model_path}")

# Power Ball
#power_model_path = "backend/gan_powerball_generator.pth"
power_model_path = from_root("backend", "gan_powerball_generator.pth")

G_power = PowerBallGenerator(power_latent_dim, power_condition_dim, power_num_classes * 5, power_mega_classes).to(device)

if os.path.exists(power_model_path):
    G_power.load_state_dict(torch.load(power_model_path, map_location=device))
    G_power.eval()
    print("Power Ball 生成器模型加载成功！")
else:
    raise FileNotFoundError(f"未找到模型文件：{power_model_path}")

# **推理函数**
def generate_numbers(generator, batch_size, latent_dim, condition_features, num_classes, mega_classes):
    """
    通用推理函数，用于生成彩票号码。

    Args:
        generator (torch.nn.Module): 生成器模型。
        batch_size (int): 生成号码的组数。
        latent_dim (int): 随机噪声维度。
        condition_features (torch.Tensor): 条件特征张量。
        num_classes (int): 主号码范围。
        mega_classes (int): Mega Ball 范围。

    Returns:
        tuple: 主号码和 Mega Ball 的生成结果。
    """
    with torch.no_grad():
        noise = torch.randn(batch_size, latent_dim).to(device)
        condition_sample = condition_features.repeat(batch_size, 1).to(device)
        numbers, mega = generator(noise, condition_sample)

        def post_process_unique(numbers):
            batch_size, _, num_classes = numbers.size()
            result = torch.zeros_like(numbers)
            for i in range(batch_size):
                chosen = set()
                for j in range(5):
                    idx = torch.argmax(numbers[i, j]).item()
                    while idx in chosen:
                        numbers[i, j, idx] = 0
                        idx = torch.argmax(numbers[i, j]).item()
                    chosen.add(idx)
                    result[i, j, idx] = 1
            return result

        numbers = post_process_unique(numbers)
        generated_numbers = torch.argmax(numbers, dim=-1).cpu().numpy() + 1
        generated_mega = torch.argmax(mega, dim=-1).cpu().numpy() + 1

        return generated_numbers.tolist(), generated_mega.tolist()


# **API 路由**
@app.route("/generate/mega_millions", methods=["GET"])
@cross_origin()
def generate_mega_millions():
    batch_size = int(request.args.get("batch_size", 1))  # 从请求参数中获取批次大小
    generated_numbers, generated_mega = generate_numbers(
        G_mega, batch_size, mega_latent_dim, mega_condition_features, mega_num_classes, mega_mega_classes
    )

    # 将生成的数字组合成前端期望的格式
    results = [
        {"numbers": generated_numbers[i], "mega_ball": generated_mega[i]} for i in range(batch_size)
    ]
    print("Generated results:", results)  # 打印调试信息

    return jsonify({"status": "success", "results": results})


@app.route("/generate/power_ball", methods=["GET"])
@cross_origin()
def generate_power_ball():
    batch_size = int(request.args.get("batch_size", 1))  # 从请求参数中获取批次大小
    generated_numbers, generated_mega = generate_numbers(
        G_power, batch_size, power_latent_dim, power_condition_features, power_num_classes, power_mega_classes
    )

    # 将生成的数字组合成前端期望的格式
    results = [
        {"numbers": generated_numbers[i], "power_ball": generated_mega[i]} for i in range(batch_size)
    ]
    print("Generated results:", results)  # 打印调试信息

    return jsonify({"status": "success", "results": results})


# # **API 路由**
# @app.route("/generate/mega_millions", methods=["GET"])
# def generate_mega_millions():
#     generated_numbers, generated_mega = generate_numbers(
#         G_mega, 1, mega_latent_dim, mega_condition_features, mega_num_classes, mega_mega_classes
#     )
#     print({"numbers": generated_numbers, "mega_ball": generated_mega})  # 打印调试信息
#     return jsonify({
#         "results": {
#             "numbers": generated_numbers,
#             "mega_ball": generated_mega
#         }
#         # "numbers": generated_numbers, "mega_ball": generated_mega
#     })
#
# @app.route("/generate/power_ball", methods=["GET"])
# def generate_power_ball():
#     generated_numbers, generated_mega = generate_numbers(
#         G_power, 1, power_latent_dim, power_condition_features, power_num_classes, power_mega_classes
#     )
#     print({"numbers": generated_numbers, "power_ball": generated_mega})  # 打印调试信息
#     return jsonify({
#         "results": {
#             "numbers": generated_numbers,
#             "mega_ball": generated_mega
#         }
#         # "numbers": generated_numbers, "power_ball": generated_mega
#     })




# # **根路径路由**
# @app.route("/", methods=["GET"])
# def home():
#     return """
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <title>Lottery Number Generator</title>
#     </head>
#     <body>
#         <h1>Welcome to the Lottery Number Generator API</h1>
#         <p>Use the form below to generate lottery numbers:</p>
#         <form action="/generate" method="post" enctype="application/json">
#             <label for="batch_size">Batch Size (1-10):</label>
#             <input type="number" id="batch_size" name="batch_size" min="1" max="10" required>
#             <button type="submit">Generate</button>
#         </form>
#     </body>
#     </html>
#     """
#
#
# # API 路由
# @app.route("/generate", methods=["POST"])
# def generate():
#     try:
#         # 检查是否来自表单提交
#         if request.form:
#             batch_size = int(request.form.get("batch_size", 1))
#         else:  # 默认解析 JSON
#             batch_size = int(request.json.get("batch_size", 1))
#
#         if batch_size < 1 or batch_size > 10:
#             return jsonify({"error": "batch_size 应在 1 到 10 之间"}), 400
#
#         numbers, mega = generate_numbers(batch_size)
#         results = [
#             {"numbers": numbers[i], "mega_ball": mega[i]} for i in range(batch_size)
#         ]
#         # 打印 results 到控制台
#         print("Generated results:", results)
#
#         return jsonify({"status": "success", "results": results})
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500





# **历史 Mega Million 数据 API**
@app.route('/api/history-numbers/MegaMillion', methods=['GET'])
@cache.cached(timeout=3600)
def get_mega_millions_data():
    try:
        data = fetch_mega_millions_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# **历史 Mega Million Statistic 数据 API**
@app.route('/api/history-statistic/MegaMillion', methods=['GET'])
@cache.cached(timeout=3600)  # 缓存 1 小时
def get_mega_million_statistic_data():
    try:
        statistics_data = fetch_mega_millions_statistic_data()
        return jsonify(statistics_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500





# **历史 Power 数据 API**
@app.route('/api/history-numbers/PowerBall', methods=['GET'])
@cache.cached(timeout=3600)
def get_power_ball_data():
    try:
        data = fetch_power_ball_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# **历史 Power 数据 Statistic API**
@app.route('/api/history-statistic/PowerBall', methods=['GET'])
@cache.cached(timeout=3600)  # 缓存 1 小时
def get_power_ball_statistic_data():
    try:
        statistics_data = fetch_power_ball_statistic_data()
        return jsonify(statistics_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# **启动服务**
if __name__ == "__main__":
   # app.run(host="0.0.0.0", port=5000)
    port = int(os.environ.get("PORT", 5000))  # Render会自动设置PORT环境变量
    app.run(host="0.0.0.0", port=port)
