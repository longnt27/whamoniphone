import torch
import torch.nn as nn
import timm
import coremltools as ct

class FastViTStudent(nn.Module):
    def __init__(self, model_name='fastvit_sa24', target_dim=1024):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256)
            student_out_dim = self.backbone(dummy_input).shape[1]

        self.proj = nn.Linear(student_out_dim, target_dim)

    def forward(self, x):
        features = self.backbone(x)
        return self.proj(features)

model = FastViTStudent(target_dim=1024).to('cpu')

model_weights_path = 'fastvit_student_1024.pth'
model.load_state_dict(torch.load(model_weights_path, map_location='cpu'))
model.eval()

example_input = torch.rand(1, 3, 256, 256)

traced_model = torch.jit.trace(model, example_input)

mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(shape=example_input.shape, name="image_input")],
    outputs=[ct.TensorType(name="features_1024")],
    minimum_deployment_target=ct.target.iOS16
)

mlmodel.save("FastViT_WHAM_1024.mlpackage")
