import dspy
from dspy.teleprompt import LabeledFewShot
from dspy.adapters import ChatAdapter
from audio_assistant import AudioAssistant
# from ModelTrain.dp.utils import get_config
import textwrap
import yaml
import time
import glob
import json
import os
import re


def get_config(config_path=None):
    if config_path is None:
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_file_dir, 'configs/configs.yaml')
    assert config_path and os.path.exists(config_path), f'configs file does not exist ({config_path})'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


class ECOTAdapter(ChatAdapter):
    """
    Adapter for generating structured ECOT task descriptions based on a system prompt.
    """
    def __init__(self, system_prompt):
        super().__init__()
        self.system_prompt = system_prompt

    def format_task_description(self, signature):
        # 使用textwrap.dedent()函数去除字符串中的缩进
        instructions = textwrap.dedent(self.system_prompt)
        # 将字符串按行分割，并在每行前添加8个空格
        objective = ("\n" + " " * 8).join([""] + instructions.splitlines())
        # 返回格式化后的任务描述
        return f"In adhering to this structure, your objective is: {objective}"


class GenerateECOTReasoning(dspy.Signature):

    feedback: str = dspy.InputField(desc="Language feedback from the user")
    keypoints = dspy.InputField(desc="3D keypoint positions of target objects and robot end-effectors")
    scene_image: dspy.Image = dspy.InputField(desc="RGB image of the manipulation scene")
    ecot_reasoning: str = dspy.OutputField(desc="Step-by-step embodied chain-of-thought reasoning")
    bimanual_category: str = dspy.OutputField(
        desc="Category of bimanual manipulation (uni_l, uni_r, uncoord_bi, asym_l_dom, asym_r_dom, sym)")
    reward_code = dspy.OutputField(desc="None-differential black box reward code that guides the robot's "
                                        "action")


class EmbodiedCoTReasoner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(GenerateECOTReasoning)

    def forward(self, feedback, keypoints, scene_image):
        prediction = self.predict(feedback=feedback, keypoints=keypoints, scene_image=scene_image)
        return dspy.Prediction(
            ecot_reasoning=prediction.ecot_reasoning,
            bimanual_category=prediction.bimanual_category,
            reward_code=prediction.reward_code
        )


class BimanualEcotReasoning:
    def __init__(self, config, example_path):
        self.example_path = example_path
        self.config = get_config(config_path=config)
        self.api_key = self.config['api_key']
        self.base_url = self.config['base_url']

        # initialize audio assistant
        self.audio_model = self.config['audio_assistant']['model']
        self.user_input_filename = self.config['audio_assistant']['user_input_filename']
        self.audio_assistant = AudioAssistant(self.audio_model, self.api_key, self.base_url, self.user_input_filename)

        # initialize LLM
        self.model = self.config['mllm']['model']
        self.temperature = self.config['mllm']['temperature']
        self.max_tokens = self.config['mllm']['max_tokens']
        self.keypoints_pth = self.config['mllm']['keypoints']
        self.scene_image_pth = self.config['mllm']['scene_img']

        # inference output
        self.feedback = None
        self.reasoning = None
        self.bimanual_category = None
        self.reward_function = None


    def load_examples(self):
        # Load language feedback
        feedback_path = os.path.join(self.example_path, 'feedback.txt')
        with open(feedback_path, 'r') as f:
            feedback = f.read()
        feedback = feedback.split('\n\n')

        # Load keypoints
        keypoints_paths = [os.path.join(self.example_path, 'keypoints', f'keypoints_{i}.json') for i in range(len(
            feedback))]
        keypoints_list = []
        for keypoints_path in keypoints_paths:
            with open(keypoints_path, 'r') as f:
                keypoints = json.load(f)
                keypoints_list.append(keypoints)

        # Load scene images
        image_paths = [os.path.join(self.example_path, 'scene_image', f'scene_{i}.png') for i in range(len(feedback))]
        scene_images = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                scene_images.append(dspy.Image.from_file(img_path))
            else:
                print("Warning: Scene image not found at", img_path)

        # Load ecot reasoning
        reasoning_path = os.path.join(self.example_path,'' ,'reasoning.txt')
        with open(reasoning_path, 'r') as f:
            reasoning_examples = f.read()
        reasoning_examples = reasoning_examples.split('\n\n')

        # Load bimanual categories
        categories_path = os.path.join(self.example_path, 'categories.txt')
        with open(categories_path, 'r') as f:
            categories = f.read()
        categories = categories.split('\n')

        # Load NBCFs reward functions
        reward_folder = os.path.join(self.example_path, 'reward')
        reward_paths = glob.glob(os.path.join(reward_folder, 'reward_*.py'))
        reward_paths = sorted(
            reward_paths,
            key=lambda x: int(re.search(r'reward_(\d+)\.py', os.path.basename(x)).group(1))
        )
        rewards = []
        for reward_path in reward_paths:
            with open(reward_path, 'r') as f:
                reward = f.read()
            rewards.append(reward)

        assert len(feedback) == len(keypoints_list) == len(reasoning_examples) == len(categories) == len(rewards), \
            f'Different number of examples in files'

        # Build examples
        examples = []
        for i in range(len(feedback)):
            examples.append(dspy.Example(
                feedback=feedback[i],
                keypoints=keypoints_list[i],
                scene_image=scene_images[i],
                ecot_reasoning=reasoning_examples[i],
                bimanual_category=categories[i],
                reward_code=rewards[i]
            ))
        return examples


    def build_llm_module(self):
        # Load system prompt
        system_prompt_path = os.path.join(self.example_path, 'system_prompt.txt')
        if os.path.exists(system_prompt_path):
            with open(system_prompt_path, 'r') as f:
                system_prompt = f.read()
            self.system_prompt = system_prompt.strip()

        # Configure LM
        self.adapter = ECOTAdapter(system_prompt=self.system_prompt)
        self.llm = dspy.LM(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            base_url=self.base_url,
        )

        dspy.settings.configure(lm=self.llm, adapter=self.adapter)

        # Load examples
        examples = self.load_examples()

        # Build and compile module
        llm_module = EmbodiedCoTReasoner()
        tp = LabeledFewShot(k=len(examples))
        self.llm_module = tp.compile(llm_module, trainset=examples)


    def generate_ecot_reasoning(self):
        """
        Generate embodied chain-of-thought reasoning from language feedback, scene image, and keypoints
        """
        # Invoke LLM here
        self.build_llm_module()

        # process inputs
        keypoints = json.dumps(self.keypoints_pth)
        scene_image_obj = None
        if self.scene_image_pth is not None:
            scene_image_obj = dspy.Image.from_file(self.scene_image_pth)

        # run the audio assistant in interactive mode
        self.audio_assistant.record_audio()
        self.feedback = self.audio_assistant.transcribe_audio()

        # process outputs
        output = self.llm_module(feedback=self.feedback, keypoints=keypoints, scene_image=scene_image_obj)

        self.reasoning = output.ecot_reasoning
        self.bimanual_category = output.bimanual_category
        self.reward_code = output.reward_code

        print(f'LANGUAGE FEEDBACK: {self.feedback}')
        # print(f'KEYPOINTS: {self.keypoints}')
        print(f'ECOT REASONING: {self.reasoning}')
        print(f'BIMANUAL CATEGORY: {self.bimanual_category}')
        print(f'REWARD CODE:\n{self.reward_code}')

        # try:
        #     print('REWARD FUNCTION:')
        #     print(self.reward_code)
        #     # Properly format the reward function code
        #     reward_code_indented = "\n".join(f"    {line}" for line in self.reward_code.split("\n"))
        #     reward_fn_code = f"def reward_fn(self, trajectory):\n{reward_code_indented}"
        #
        #     reward_code_namespace = {}
        #     exec(reward_fn_code, globals(), reward_code_namespace)
        #     self.reward_function = reward_code_namespace['reward_fn']
        # except Exception:
        #     print('*' * 40)
        #     print('Exception encountered while executing LLM code, trace below')
        #     print('*' * 40)
        #     print(traceback.format_exc())
        #     print('*' * 40)
        #     import pdb;
        #     pdb.set_trace()

        return {
            'reasoning': self.reasoning,
            'bimanual_category': self.bimanual_category,
            'reward_function': self.reward_code
        }


if __name__ == '__main__':
    # Example usage
    example_path = '../assets/ecot_prompt'
    config = "../configs/llm_config.yaml"
    ecot_reasoner = BimanualEcotReasoning(config=config, example_path=example_path)
    time1 = time.time()
    result = ecot_reasoner.generate_ecot_reasoning()
    time2 = time.time()

    print(f'Time taken: {time2 - time1:.2f} seconds')