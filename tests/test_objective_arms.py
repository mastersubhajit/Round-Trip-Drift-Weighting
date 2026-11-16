"""CPU tests for the proposed anti-poison arms: V (all-pairs vaccine) and R (DCLR latent re-grounding)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
import train_full_unet as T  # noqa: E402
import dclr_latents as DC  # noqa: E402


class FakeSched:
    config = SimpleNamespace(num_train_timesteps=10, prediction_type='epsilon')
    def add_noise(self, x, e, t): return x + 0.1 * e


class FakeVAE:
    config = SimpleNamespace(scaling_factor=1.0)
    def encode(self, x): raise AssertionError('encoder must not be called when latents are supplied')


class FakeTok:
    model_max_length = 4
    def __call__(self, caps, **k):
        ids = torch.tensor([[hash(w) % 50 for w in (c.split() + ['_'] * 4)[:4]] for c in caps])
        return SimpleNamespace(input_ids=ids)


class FakeEnc(torch.nn.Module):
    def __init__(self): super().__init__(); self.e = torch.nn.Embedding(50, 3)
    def forward(self, ids): return (self.e(ids),)


class FakeUNet(torch.nn.Module):
    """Prediction depends on the caption embedding, so caption swaps change the loss."""
    def __init__(self): super().__init__(); self.w = torch.nn.Parameter(torch.randn(3, 2))
    def forward(self, z, t, h):
        return SimpleNamespace(sample=z * 0 + (h.mean(1) @ self.w).sum(-1).view(-1, 1, 1, 1))


def test_rival_captions_cover_all_other_concepts_and_fail_closed():
    r = T.rival_captions('a cat on a sofa', 'cat', ['dog', 'cat', 'car'])
    assert r == ['a dog on a sofa', 'a car on a sofa']
    with pytest.raises(ValueError):
        T.rival_captions('a sofa', 'cat', ['dog', 'cat'])


def test_vaccine_margin_includes_target_direction_and_backpropagates():
    torch.manual_seed(0)
    unet, enc = FakeUNet(), FakeEnc()
    px = torch.zeros(2, 3, 8, 8)
    vae = SimpleNamespace(config=SimpleNamespace(scaling_factor=1.0),
                          encode=lambda x: SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: torch.zeros(2, 4, 1, 1))))
    pos, margin, n = T.vaccine_margin_losses(unet, FakeSched(), vae, enc, FakeTok(), px,
                                             ['a cat here', 'a dog here'], ['cat', 'dog'], ['dog', 'cat', 'car'],
                                             torch.float32, False, 0.1)
    assert n == 2 and torch.isfinite(margin) and margin > 0
    margin.backward()
    assert unet.w.grad is not None and unet.w.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='not in the vaccine vocabulary'):
        T.vaccine_margin_losses(unet, FakeSched(), vae, enc, FakeTok(), px, ['a cow', 'a dog'], ['cow', 'dog'],
                                ['dog', 'cat'], torch.float32, False, 0.1)


def test_supplied_latents_bypass_encoder():
    z = torch.ones(2, 4, 2, 2)
    noised, t, target = T.encode_pixels(FakeVAE(), FakeSched(), torch.zeros(2, 3, 16, 16), torch.float32, latents=z)
    assert noised.shape == z.shape and target.shape == z.shape


def test_latent_cache_validation(tmp_path):
    d = tmp_path / 'train'; d.mkdir()
    (d / 'metadata.jsonl').write_text(''.join(json.dumps({'id': i, 'file_name': f'{i}.png', 'text': 'x'}) + '\n' for i in 'ab'))
    f = tmp_path / 'lat.pt'
    torch.save({'a': torch.zeros(4, 8, 8), 'b': torch.zeros(4, 8, 8)}, f)
    assert set(T.load_latent_cache(f, d, 64)) == {'a', 'b'}
    torch.save({'a': torch.zeros(4, 8, 8)}, f)
    with pytest.raises(ValueError, match='lacks 1'):
        T.load_latent_cache(f, d, 64)
    torch.save({'a': torch.zeros(4, 8, 8), 'b': torch.full((4, 8, 8), float('nan'))}, f)
    with pytest.raises(ValueError):
        T.load_latent_cache(f, d, 64)


def test_dclr_recovers_pixel_consistent_latent_when_encoder_is_fooled():
    """Toy version of the anti-poison argument: a linear decoder D, an encoder fooled by a small pixel
    perturbation into a 'target' latent; the decoder pre-image of the poisoned pixels stays near the
    source latent, far from the target latent."""
    torch.manual_seed(0)
    Dm = torch.nn.Linear(4, 12, bias=False)
    class Dec:
        def decode(self, z): return SimpleNamespace(sample=Dm(z.flatten(1)).view(-1, 12, 1, 1))
    z_src, z_tgt = torch.randn(1, 4), torch.randn(1, 4) * 3
    x_src = Dm(z_src).detach().view(1, 12, 1, 1)
    x_poison = x_src + 0.01 * torch.randn_like(x_src)          # stealthy: pixels ~ source
    z_enc_poison = z_tgt.view(1, 4, 1, 1)                     # fooled encoder output (attack succeeded on E)
    z = DC.solve(Dec(), x_poison, z_enc_poison.view(1, 4), steps=600, lr=0.05, beta=0.0, autocast_dtype=None)
    assert (z - z_src).norm() < 0.2 * (z_enc_poison.view(1, 4) - z_src).norm()
    assert ((Dec().decode(z).sample - x_poison) ** 2).mean() < ((Dec().decode(z_enc_poison.view(1, 4)).sample - x_poison) ** 2).mean()


class LinDecoder:
    """Tiny decoder: D(z) = tanh(conv(z)), enough to test the DCLR-S objective on CPU."""
    def __init__(self):
        torch.manual_seed(1); self.c = torch.nn.Conv2d(4, 3, 1)
        for p in self.c.parameters(): p.requires_grad_(False)
    def decode(self, z): return SimpleNamespace(sample=torch.tanh(self.c(z)))


def _sem(x):  # differentiable toy embedder: per-channel means, unit-normalised
    e = x.mean((2, 3)) + torch.tensor([0.3, 0.0, 0.0]); return e / e.norm(dim=-1, keepdim=True)


def test_dclr_s_gamma_zero_matches_original_objective():
    vae = LinDecoder(); x = torch.rand(2, 3, 4, 4) * 2 - 1; z0 = torch.randn(2, 4, 4, 4)
    a = DC.solve(vae, x, z0, 5, 0.05, 0.1, None)
    b = DC.solve(vae, x, z0, 5, 0.05, 0.1, None, sem=_sem, gamma=0.0)
    assert torch.allclose(a, b)


def test_dclr_s_semantic_term_reduces_semantic_distance():
    vae = LinDecoder(); x = torch.rand(2, 3, 4, 4) * 2 - 1; z0 = torch.randn(2, 4, 4, 4)
    dist = lambda z: (1 - (_sem(vae.decode(z).sample.clamp(-1, 1)) * _sem(x)).sum(-1)).mean().item()
    plain = DC.solve(vae, x, z0, 30, 0.05, 0.0, None)
    sem = DC.solve(vae, x, z0, 30, 0.05, 0.0, None, sem=_sem, gamma=10.0)
    assert dist(sem) < dist(plain)


def test_clip_preprocess_shape_and_normalisation():
    out = DC.clip_preprocess(torch.zeros(1, 3, 512, 512))   # mid-grey 0.5
    assert out.shape == (1, 3, 224, 224)
    exp = (0.5 - torch.tensor(DC.CLIP_MEAN)) / torch.tensor(DC.CLIP_STD)
    assert torch.allclose(out[0, :, 0, 0], exp, atol=1e-5)
