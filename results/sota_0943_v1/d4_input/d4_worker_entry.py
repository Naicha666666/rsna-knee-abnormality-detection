import sys
sys.path.insert(0,'/kaggle/working/_d4_env')
sys.path.insert(0,'/kaggle/input/datasets/mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2')
from pathlib import Path
import json
import coatnet_d4_depthzone_swa_inference as rt
def _asset_patch_d4(rt, grid_source):
    import ast
    import hashlib
    import inspect
    import json
    import os
    import textwrap
    from pathlib import Path
    audited = rt.parent.audited
    original = audited.prepare_global96_bag
    original_shard = rt._PARENT_INFER_SHARD
    if getattr(original, '_input_patch_installed', False):
        raise RuntimeError('D4 input preparation was already patched')
    src = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(src)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fn is None or fn.name != 'prepare_global96_bag':
        raise RuntimeError('D4 preparation function identity changed')
    if 'study_uid' not in inspect.signature(original).parameters:
        raise RuntimeError('D4 preparation study-UID contract changed')
    candidates = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'specs' for t in n.targets)) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name) and (n.value.func.value.id == 'global_stack') and (n.value.func.attr == 'sample_global_windows')]
    if len(candidates) != 1 or candidates[0] not in fn.body:
        raise RuntimeError('D4 expected one top-level Global96 specs constructor')
    call = candidates[0]
    keywords = {k.arg: k.value for k in call.value.keywords}
    if not (isinstance(keywords.get('count'), ast.Constant) and keywords['count'].value is None and isinstance(keywords.get('train'), ast.Constant) and (keywords['train'].value is False)):
        raise RuntimeError('D4 original evaluation must use count=None, train=False')
    exec(compile(grid_source, '<input-grid>', 'exec'), audited.__dict__)
    audit_dir = Path(os.environ['RSNA_D4_INPUT_AUDIT_DIR'])
    audit_dir.mkdir(parents=True, exist_ok=True)

    def record_input(study_uid, stack, audit):
        import numpy as np
        arrays = ('global_indices', 'nominal_slots', 'canonical_depths', 'nominal_steps', 'source_series_rows', 'source_slot_ids')
        digest = hashlib.sha256()
        for name in arrays:
            value = np.ascontiguousarray(getattr(stack, name))
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        record = {'study_uid': str(study_uid), 'pid': os.getpid(), 'positions': int(len(stack.global_indices)), 'windows_expected': int(len(stack.global_indices) - 2), 'slot_widths': list(map(int, stack.slot_widths)), 'stack_sidecars_sha256': digest.hexdigest(), 'slots': audit}
        with (audit_dir / f'input_{os.getpid()}.jsonl').open('a') as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
    audited.__dict__['_record_input'] = record_input
    injection = ast.parse('try:\n    stack, input_audit = _dense_stack(stack, study.offsets, study.valid, study.source_depth)\n'
                          'except Exception as _dense_exc:\n    print("[dense-fallback] " + str(study_uid) + ": " + type(_dense_exc).__name__ + ": " + str(_dense_exc), flush=True)\n'
                          '    input_audit = [{"fallback": type(_dense_exc).__name__ + ": " + str(_dense_exc)}]\n'
                          '_record_input(study_uid, stack, input_audit)\n').body
    where = fn.body.index(call)
    fn.body[where:where] = injection
    ast.fix_missing_locations(tree)
    exec(compile(tree, '<D4-input-preparation-only>', 'exec'), audited.__dict__)
    patched = audited.prepare_global96_bag
    patched._input_patch_installed = True
    for module in (rt, rt.parent):
        if getattr(module, 'prepare_global96_bag', None) is original:
            module.prepare_global96_bag = patched
    if rt._PARENT_INFER_SHARD is not original_shard:
        raise RuntimeError('D4 parent shard must remain the official callable')
    files = {}
    for name, module in [('d4_runtime', rt), ('parent_runtime', rt.parent), ('audited_runtime', audited)]:
        path = Path(module.__file__)
        if path.is_file():
            files[name] = {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    receipt = {'contract': 'd4-original-runtime-input-preparation-only-v2', 'reference_script_version_id': 348764100, 'reference_export_available': True, 'original_prepare_source_sha256': hashlib.sha256(src.encode()).hexdigest(), 'patched_prepare_source_sha256': hashlib.sha256(ast.unparse(tree).encode()).hexdigest(), 'model_loader_modified': False, 'zone_head_modified': False, 'parent_infer_shard_replaced': False, 'grid_source_sha256': hashlib.sha256(grid_source.encode()).hexdigest(), 'artifact_source_files': files}
    (audit_dir / f'patch_{os.getpid()}.json').write_text(json.dumps(receipt, indent=2))
    return receipt

def _d4_check_runtime(rt, artifact_root):
    import hashlib
    import json
    from pathlib import Path
    import torch
    root = Path(artifact_root)
    manifest_path = root / 'coatnet_pairfilm_manifest.json'
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest != '7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d':
        raise RuntimeError('D4 reference manifest identity changed')
    expected = json.loads(manifest_path.read_text())
    manifest, paths = rt.validate_artifact_root(root, verify_hashes=True)
    if manifest != expected or len(paths) != 1 or rt.N_MODELS != 1:
        raise RuntimeError('D4 must load exactly the reference SWA model')
    if Path(paths[0]).name != 'coatnet_d4_parent_rank8_swa2.pt':
        raise RuntimeError('D4 must use its reference rank-8 parent checkpoint')
    if manifest['slice_depth']['adapter_sha256'] != 'b7c48b19997bc7ecb7e997f8dcf491ec5701ff917d1dac4b369eb85ac97ae89a':
        raise RuntimeError('D4 reference adapter identity changed')
    if rt.GPU_BATCH_STUDIES != 2 or rt.BACKBONE_MICRO_IMAGES != 8:
        raise RuntimeError('D4 reference batching changed')
    if rt.CUDNN_BENCHMARK is not False:
        raise RuntimeError('D4 reference convolution policy changed')
    if rt.base.cv2.__version__ != '4.12.0' or rt.parent.timm.__version__ != '1.0.22':
        raise RuntimeError('D4 reference dependency versions changed')
    if torch.cuda.device_count() != 2:
        raise RuntimeError('D4 requires exactly two T4 GPUs')
    names = [torch.cuda.get_device_name(i) for i in range(2)]
    if not all(('T4' in name for name in names)):
        raise RuntimeError(f'D4 reference device mismatch: {names}')
    return (manifest, paths)

def _d4_check_outputs(rt, manifest, receipt, output, competition):
    import hashlib
    from pathlib import Path
    import numpy as np
    import pandas as pd

    def need(condition, message):
        if not condition:
            raise RuntimeError('D4 release check: ' + message)
    need(receipt['status'] == rt.SUBMISSION_STATUS, 'unsuccessful status')
    expected_fields = {'models': 1, 'fallback_studies': 0, 'dicom_preparations_per_study': 1, 'model_passes_per_study': 1, 'rank_after_probability_average': True, 'canonical_sagittal': True, 'common_physical_triplet_warp': False, 'resolution': 384, 'slice_depth_arm': 'd4_zones', 'slice_depth_state_elements': 3255, 'slice_depth_parent_checkpoint_sha256': '5de38e333e3184d7b52fb068183f0cc198d6cbabb6ffbe36b26338622e82d297', 'fsx_attention_heads': 2, 'fsx_attention_width': 128, 'fsx_serving_delta_cap': 0.3, 'fsx_training_delta_cap': 0.5, 'head': 'rank8_swa2_plus_d4_three_zone_expanded_fsx', 'slice_depth_swa_member_epochs': [11, 9, 5], 'slice_depth_checkpoint_sha256': manifest['slice_depth']['adapter_sha256']}
    for key, value in expected_fields.items():
        need(receipt[key] == value, key)
    need(1 <= int(receipt['maximum_eval_windows']) <= 94, 'window limit')
    checkpoints = receipt['checkpoints']
    need(len(checkpoints) == 1, 'parent checkpoint count')
    need(checkpoints[0]['sha256'] == '5de38e333e3184d7b52fb068183f0cc198d6cbabb6ffbe36b26338622e82d297', 'parent checkpoint hash')
    need(checkpoints[0]['member_epochs'] == [15, 16], 'parent SWA member epochs')
    need(receipt['slice_depth_checkpoint_sha256'] == 'b7c48b19997bc7ecb7e997f8dcf491ec5701ff917d1dac4b369eb85ac97ae89a', 'adapter checkpoint hash')
    processes = receipt['processes']
    need(len(processes) == 2 and all((int(p['returncode']) == 0 for p in processes)), 'worker process failure')
    shards = receipt['shards']
    need(len(shards) == 2, 'shard count')
    for s in shards:
        need(s['device'] == 'cuda', 'non-CUDA shard')
        need(int(s['fallback_studies']) == 0 and int(s['preparation_warnings']) == 0, 'failed study/preparation')
        need(int(s['models_resident']) == 1, 'resident model count')
        need(int(s['peak_reserved_bytes']) < int(15.5 * 1024 ** 3), 'reference memory gate')
        need(s['specialized_worker_loader_contract'] == manifest['worker_loader_contract'], 'specialized loader')
    output = Path(output)
    prediction_path = output.parent / rt.PREDICTIONS_NAME
    with np.load(prediction_path, allow_pickle=False) as payload:
        uids = payload['study_uids'].astype(str).tolist()
        raw = np.asarray(payload['raw_probabilities'], dtype=np.float32)
        mean = np.asarray(payload['probability_mean'], dtype=np.float32)
        ranked = np.asarray(payload['submission_percentile_rank'], dtype=np.float64)
    ids = pd.read_csv(Path(competition) / 'test.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
    need(len(ids) > 0 and len(ids) == len(set(ids)), 'test identity')
    need(uids == ids and int(receipt['studies']) == len(ids), 'prediction identity/count')
    need(raw.shape == (1, len(ids), 12), 'raw probability shape')
    need(np.isfinite(raw).all() and ((raw >= 0) & (raw <= 1)).all(), 'raw probabilities')
    need(np.array_equal(raw.astype(np.float64).mean(0).astype(np.float32), mean), 'probability mean')
    need(np.array_equal(rt.percentile_rank64(mean), ranked), 'global percentile ranks')
    frame = pd.read_csv(output, dtype={'StudyInstanceUID': str}, float_precision='round_trip')
    need(frame.columns.tolist() == ['StudyInstanceUID', *rt.base.LABELS], 'CSV labels')
    need(frame.StudyInstanceUID.tolist() == ids, 'CSV identity')
    need(np.array_equal(frame.iloc[:, 1:].to_numpy(np.float64), ranked), 'CSV numeric round trip')
    actual_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    need(actual_hash == receipt['output_sha256'], 'CSV hash')
    receipt['reference_wrapper_checks'] = {'export_sha256': '7f8410fdf1b60cd3831d50f517bb25798a4e4a3404c59dc269193d5fe7cce729', 'validated_manifest_sha256': '7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d', 'probability_mean_exact': True, 'ranks_exact': True, 'csv_roundtrip_exact': True, 'source_and_model_hashes_validated': True, 'input_identity_equals_original_grid': False}
    return receipt

if __name__ == '__main__' and '--worker-output' not in sys.argv:
    _checked_manifest, _checked_paths = _d4_check_runtime(rt, Path('/kaggle/input/datasets/mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2'))
_asset_patch_d4(rt,"import numpy as np\nfrom raptor_global_stack_smart336 import FixedGlobalStack, INFERRED96_WIDTHS\ndef _dense_quotas(capacities) -> np.ndarray:\n    capacities = np.asarray(capacities, dtype=np.int64)\n    base = np.asarray(INFERRED96_WIDTHS, dtype=np.int64)\n    if capacities.shape != (5,) or bool((capacities < 0).any()):\n        raise ValueError('expected five nonnegative source capacities')\n    quotas = np.minimum(base, capacities)\n    target = min(96, int(capacities.sum()))\n    deficit = np.zeros(5, dtype=np.float64)\n    while int(quotas.sum()) < target:\n        donors = quotas < capacities\n        weights = np.where(donors, base, 0).astype(np.float64)\n        deficit[donors] += weights[donors] / weights.sum()\n        selected = int(np.argmax(np.where(donors, deficit, -np.inf)))\n        quotas[selected] += 1\n        deficit[selected] -= 1\n    if int(quotas.sum()) != target or bool((quotas > capacities).any()):\n        raise RuntimeError('additional quota allocation changed')\n    return quotas\n\ndef _dense_stack(reference, offsets, valid, source_depth):\n    from types import SimpleNamespace\n    from raptor_global_stack_smart336 import FixedGlobalStack\n    if tuple(reference.slot_widths) != tuple(INFERRED96_WIDTHS):\n        raise ValueError('capacity-aware sampling is pinned to the Global96 reference layout')\n    pools = []\n    audit = []\n    begin = 0\n    for slot, width in enumerate(reference.slot_widths, 1):\n        end = begin + width\n        old = reference.global_indices[begin:end]\n        row = int(reference.source_series_rows[begin])\n        info = {'slot': slot, 'original_width': width, 'source_row': row}\n        audit.append(info)\n        if row < 0 or bool((old < 0).all()):\n            pools.append((np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)))\n            begin = end\n            continue\n        start, stop = map(int, offsets[row:row + 2])\n        if not 0 <= start < stop <= len(valid) == len(source_depth):\n            raise ValueError('source offset drift')\n        depth = np.asarray(source_depth[start:stop], dtype=np.float32)\n        keep = np.asarray(valid[start:stop], dtype=bool) & np.isfinite(depth) & (depth >= 0.02) & (depth <= 0.98)\n        eligible = np.flatnonzero(keep).astype(np.int64) + start\n        reverse = bool(old[0] > old[-1] or (old[0] == old[-1] and np.float32(source_depth[old[0]]) != reference.canonical_depths[begin]))\n        if reverse:\n            eligible = eligible[::-1].copy()\n        depth = np.asarray(source_depth[eligible], dtype=np.float32)\n        if reverse:\n            depth = (1.0 - depth).astype(np.float32)\n        pools.append((eligible, depth))\n        begin = end\n    quotas = _dense_quotas([len(pool[0]) for pool in pools])\n    if int(quotas.sum()) < 3:\n        raise ValueError('study has fewer than three usable unique source slices')\n    parts = {key: [] for key in ('global_indices', 'nominal_slots', 'canonical_depths', 'nominal_steps', 'source_series_rows', 'source_slot_ids')}\n    for slot, ((eligible, depth), quota, info) in enumerate(zip(pools, quotas, audit), 1):\n        quota = int(quota)\n        info.update(eligible=len(eligible), allocated_width=quota, additional=max(0, quota - info['original_width']))\n        info['reason'] = 'missing_no_padding' if not quota else 'short_slot_all_unique' if len(eligible) < info['original_width'] else 'uniform_native_centers' if info['additional'] else 'uniform_native_centers'\n        positions = np.linspace(0, len(eligible) - 1, quota).round().astype(np.int64)\n        if len(positions) != quota or not bool((np.diff(positions) > 0).all()):\n            raise RuntimeError('capacity-aware sampling slot selection repeated a source')\n        mask = reference.nominal_slots == slot\n        source_slot = int(reference.source_slot_ids[mask][0])\n        parts['global_indices'].append(eligible[positions])\n        parts['canonical_depths'].append(depth[positions])\n        parts['nominal_slots'].append(np.full(quota, slot, dtype=np.int8))\n        parts['nominal_steps'].append(np.full(quota, 0.96 / max(quota - 1, 1), dtype=np.float32))\n        parts['source_series_rows'].append(np.full(quota, info['source_row'], dtype=np.int32))\n        parts['source_slot_ids'].append(np.full(quota, source_slot, dtype=np.int8))\n    result = FixedGlobalStack(**{key: np.concatenate(value) for key, value in parts.items()}, slot_widths=tuple((int(value) for value in quotas)))\n    if len(np.unique(result.global_indices)) != len(result.global_indices) or bool((result.global_indices < 0).any()):\n        raise RuntimeError('capacity-aware sampling output contains a repeated/missing source')\n    return (result, audit)\n")
rt.parent.audited.__file__ = __file__
if __name__ == '__main__':
    if '--worker-output' in sys.argv:
        raise SystemExit(rt.main())
    manifest, paths = _checked_manifest, _checked_paths
    r = rt.run_submission(
        competition_root=Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'), artifact_root=Path('/kaggle/input/datasets/mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2'),
        output_path=Path('/kaggle/working/d4_input/d4.csv'), gpu_batch_studies=2, backbone_micro_images=8)
    r = _d4_check_outputs(rt, manifest, r, Path('/kaggle/working/d4_input/d4.csv'), Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'))
    Path('/kaggle/working/d4_input/d4.receipt.json').write_text(json.dumps(r,indent=2,allow_nan=False))
