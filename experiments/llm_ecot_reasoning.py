import dspy
from dspy.teleprompt import LabeledFewShot
from dspy.adapters import ChatAdapter
import textwrap
import glob
import json
import os
import re
import numpy as np
import base64
from PIL import Image
import io


class ECOTAdapter(ChatAdapter):
    """
    Adapter for generating structured ECOT task descriptions based on a system prompt.
    """
    def __init__(self, system_prompt):
        super().__init__()
        self.system_prompt = system_prompt

    def format_task_description(self, signature):
        instructions = textwrap.dedent(self.system_prompt)
        objective = ("\n" + " " * 8).join([""] + instructions.splitlines())
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
        self.config = config
        self.example_path = example_path

        # inference input
        self.language_feedback = self.config['feedback']
        self.keypoints = self.config.get('keypoints', {})
        self.scene_image = self.config.get('scene_image', None)

        # inference output
        self.reasoning = None
        self.bimanual_category = None
        self.reward_function = None


    def load_examples(self):
        # Load language feedback
        feedback_path = os.path.join(self.example_path, 'feedback.txt')
        with open(feedback_path, 'r') as f:
            feedback = f.read()
        feedback = feedback.split('\n')

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
                # Placeholder if image doesn't exist
                scene_images.append(None)

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
            model=self.config['model'],
            temperature=self.config['temperature'],
            max_tokens=self.config['max_tokens'],
            api_key=self.config['api_key'],
            base_url=self.config['base_url'],
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
        keypoints_json = json.dumps(self.keypoints)
        scene_image_obj = None
        if self.scene_image is not None:
            scene_image_obj = dspy.Image.from_file(self.scene_image)

        # process outputs
        output = self.llm_module(feedback=self.language_feedback, keypoints=keypoints_json, scene_image=scene_image_obj)

        self.reasoning = output.ecot_reasoning
        self.bimanual_category = output.bimanual_category
        self.reward_code = output.reward_code

        print(f'LANGUAGE FEEDBACK: {self.language_feedback}')
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
            'reward_function': self.reward_function
        }


if __name__ == '__main__':
    # Example usage
    example_path = '../assets/ecot_prompt'

    # Sample keypoints dict based on the image format
    keypoints = {
        "keypoint_positions": [
        [
            -0.5255318159787837,
            -0.6410613610313141,
            0.17507989435444005
        ],
        [
            -0.06437652336860511,
            -0.6607575073329838,
            0.2593555254794204
        ],
        [
            -0.08437955102591133,
            -0.664174649129029,
            0.14815580138111661
        ],
        [
            -0.1877555019387892,
            -0.6078141598882151,
            0.07507610074284776
        ],
        [
            0.12016399813826473,
            -0.590811761683398,
            0.10805210429815504
        ],
        [
            -0.1515832966173195,
            -0.528417194438881,
            0.11444241220729201
        ],
        [
            -0.013793949198795286,
            -0.5003526043348581,
            0.13383476327068244
        ],
        [
            0.12037205004461027,
            -0.49829600886419306,
            0.1274565447258993
        ],
        [
            0.006969026536109646,
            -0.4294304544767852,
            0.1840936898008454
        ],
        [
            -0.5611342554649539,
            -0.3845788288950187,
            0.16190481563697223
        ],
        [
            -0.5875303569769578,
            -0.3507587713071481,
            0.2760577511809834
        ]
    ],
    "num_keypoints": 11
    }

    config = {
        'model': 'gpt-4o',
        'temperature': 0.2,
        'max_tokens': 1000,
        'api_key': "sk-mSyK58YNPAdpDcyL5cFb693b61Ff4fD8A367539e68C5898c",
        'base_url': 'https://www.jcapikey.com/v1',
        'feedback': 'Adjust your left hand to wipe the plate properly',
        'keypoints': keypoints,
        'scene_image': '../assets/ecot_prompt/scene_image/scene_0.png'  # Path to RGB image
    }

    ecot_reasoner = BimanualEcotReasoning(config=config, example_path=example_path)
    result = ecot_reasoner.generate_ecot_reasoning()