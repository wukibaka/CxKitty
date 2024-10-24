import json
import requests
from cxapi.schema import QuestionModel
from . import SearcherBase, SearcherResp
from logger import Logger


class OllamaSearcherAPI(SearcherBase):
    """Ollama Llama3 在线答题器"""

    config: dict

    def __init__(self, **config) -> None:
        super().__init__()
        self.config = config
        self.logger = Logger("OllamaSearcherAPI")

    def invoke(self, question: QuestionModel) -> SearcherResp:
        # 将选项从 JSON 转换成人类易读的形式
        options_str = ""
        if question.options is not None:
            options_str = "\n\u9009\u9879\uff1a\n"
            if isinstance(question.options, dict):
                for k, v in question.options.items():
                    options_str += f"{k}. {v};"
            elif isinstance(question.options, list):
                for v in question.options:
                    options_str += f"{v};"

        # 生成请求内容
        prompt = self.config["prompt"].format(
            type=question.type.name,
            value=question.value,
            options=options_str,
        )

        system_prompt = self.config.get("system_prompt", "")
        if not system_prompt:
            raise ValueError("system_prompt is required in the configuration.")

        self.logger.info(
            f"\u4ece {self.config['prompt']} \u751f\u6210\u63d0\u95ee\uff1a{prompt}"
        )

        # 构造请求的 JSON 数据
        request_data = {
            "model": self.config.get("model", "llama3"),
            "prompt": f"{{\"role\": \"system\", \"content\": \"{system_prompt}\"}}\n{prompt}",
            "stream": False,
        }

        # 发出请求
        try:
            response = requests.post(
                url=f"{self.config['base_url']}/api/generate",
                headers={
                    "Content-Type": "application/json",
                },
                json=request_data,
            )

            # 检查响应状态
            response.raise_for_status()
            response_json = response.json()
            response_text = response_json.get("response", "")
        except requests.exceptions.RequestException as err:
            return SearcherResp(-500, str(err), self, question.value, None)

        # 根据不同题型处理 AI 返回的结果
        response_text = response_text.strip()

        if question.type.value == 0:  # 单选题
            for k, v in question.options.items():
                if response_text.startswith(f"{k}.") or (v in response_text):
                    response_text = v
                    break

        elif question.type.value == 1:  # 多选题
            selected_answers = [v for k, v in question.options.items() if v in response_text]
            response_text = "#".join(selected_answers)

        elif question.type.value == 2:  # 填空题
            # 假设填空题只需要返回填空的内容
            response_text = response_text

        elif question.type.value == 3:  # 判断题
            if "正确" in response_text or "是" in response_text:
                response_text = "正确"
            elif "错误" in response_text or "否" in response_text:
                response_text = "错误"

        elif question.type.value in [4, 5, 6, 7, 8, 9, 10]:  # 简答题, 名词解释, 论述题, 计算题, 其它, 分录题, 资料题
            # 对于这些题型，直接返回生成的答案
            response_text = response_text

        elif question.type.value == 11:  # 连线题
            # 假设连线题返回格式为 "A-1, B-2"
            response_text = response_text

        elif question.type.value == 13:  # 排序题
            # 假设排序题返回格式为 "1, 2, 3"
            response_text = response_text

        elif question.type.value == 14:  # 完型填空
            # 假设完型填空返回一个完整的段落
            response_text = response_text

        elif question.type.value == 15:  # 阅读理解
            # 假设阅读理解返回问题的答案
            response_text = response_text

        elif question.type.value == 18:  # 口语题
            # 假设口语题返回的是建议的回答内容
            response_text = response_text

        elif question.type.value == 19:  # 听力题
            # 假设听力题返回的是听到的内容
            response_text = response_text

        elif question.type.value == 20:  # 共用选项题
            # 假设共用选项题与单选题类似，返回选择的答案
            for k, v in question.options.items():
                if response_text.startswith(f"{k}.") or (v in response_text):
                    response_text = v
                    break

        elif question.type.value == 21:  # 测评题
            # 假设测评题返回测评的结果或建议
            response_text = response_text

        self.logger.info(f"\u8fd4\u56de\u7ed3\u679c\uff1a{response_text}")
        return SearcherResp(0, "", self, question.value, response_text)
