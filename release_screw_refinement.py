"""Publish the preselected screw experiment locally; never select on test."""
import json
import shutil

from screw_refine import ROOT, OUT, atomic_json, sha, validate_development


def release():
    complete=validate_development()
    selection=complete['selection']
    evaluation=json.loads((OUT/'evaluation.json').read_text(encoding='utf-8'))
    candidate=selection['selected']
    if evaluation['selected']!=candidate or evaluation['test_used_for_selection']:
        raise ValueError('Evaluation differs from the frozen development selection.')
    result=evaluation['results'][candidate]
    folder=ROOT/'release/screw_refinement'
    folder.mkdir(parents=True,exist_ok=True)
    destination=folder/'model.pt'
    shutil.copyfile(OUT/candidate/'model.pt',destination)
    metrics=result['threshold_metrics']
    manifest={'schema_version':1,'status':'experimental',
              'model':'release/screw_refinement/model.pt','model_sha256':sha(destination),
              'selected_candidate':candidate,'selection_sha256':sha(OUT/'selection.json'),
              'test_metrics':{'image_auroc':result['image_auroc'],
                              **{key:metrics[key] for key in ('tp','fp','fn','tn')}},
              'evaluation_note':'Selected on training-side synthetic development; previously exposed real test is exploratory.',
              'source_protocol_sha256':sha(OUT/'protocol.json')}
    atomic_json(folder/'manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
    return manifest


if __name__=='__main__':
    release()
