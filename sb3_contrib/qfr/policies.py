from sb3_contrib.common.maskable.policies import (
    MaskableActorCriticCnnPolicy,
    MaskablePolicyGradiantPolicy,
    MaskableActorCriticPolicy,
    MaskableMultiInputActorCriticPolicy,
    MaskableQFRPolicy
)

MlpPolicy = MaskableActorCriticPolicy
PGMlpPolicy = MaskablePolicyGradiantPolicy
CnnPolicy = MaskableActorCriticCnnPolicy
MultiInputPolicy = MaskableMultiInputActorCriticPolicy
QFRPolicy = MaskableQFRPolicy
