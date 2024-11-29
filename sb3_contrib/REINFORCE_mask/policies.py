from sb3_contrib.common.maskable.policies import (
    MaskableActorCriticCnnPolicy,
    MaskablePolicyGradiantPolicy,
    MaskableActorCriticPolicy,
    MaskableMultiInputActorCriticPolicy,
)

MlpPolicy = MaskableActorCriticPolicy
PGMlpPolicy = MaskablePolicyGradiantPolicy
CnnPolicy = MaskableActorCriticCnnPolicy
MultiInputPolicy = MaskableMultiInputActorCriticPolicy
