# PR #51 file disposition (Stage 0)

Reviewed head `304beb6`; base `8dd08f4`. Every changed file is listed below.

This conservative static inventory follows local imports and script references from the stepper, preflight, conformance, replay and Linode provisioning scripts. References in comments and optional branches can overestimate reachability; dynamic imports can escape it. It is a triage aid, not runtime coverage evidence. Confirm each dependency when implementing its batch. No deferred file is approved for blind cherry-pick.

| File | Disposition | Evidence / reason |
|---|---|---|
| `scripts/anatomy_v0_sweep.py` | Repair/review, Stage 1 | Referenced by scripts/warp_defined_pass.py; distinguish checkout, external data and sibling roots |
| `scripts/apm_proof_audit.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/background_corpus_index.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/build-ct-anatomy-viewer.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/build-msc-topic-prior.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/build-proof-anatomy-viewer.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/build_apm_crossdisc_pool.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/build_ct_manifest.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/build_ct_prior.py` | Repair/review, Stage 1 | Referenced by scripts/superpod-job.py; distinguish checkout, external data and sibling roots |
| `scripts/build_fable_golden.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/build_golden_paper.py` | Repair/review, Stage 1 | Referenced by scripts/dp_anatomy_html.py; distinguish checkout, external data and sibling roots |
| `scripts/build_mission_kernel.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/build_mission_prior.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/build_recognizer_registry.py` | Repair/review, Stage 1 | Referenced by scripts/build_term_prior.py; distinguish checkout, external data and sibling roots |
| `scripts/bv_comb_typer.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/c_mine_joint.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/c_vector.bb` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/cas_select.py` | Repair/review, Stage 1 | Referenced by scripts/rung3_technique.py; distinguish checkout, external data and sibling roots |
| `scripts/cascade_learn.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/check_invariants.py` | Repair/review, Stage 1 | Referenced by scripts/linode_stepper.py; distinguish checkout, external data and sibling roots |
| `scripts/cite_resolve.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/cite_resolve_check.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/clean_structure_embed.py` | Repair/review, Stage 1 | Referenced by scripts/linode_stepper.py; distinguish checkout, external data and sibling roots |
| `scripts/coapp_live_usage_miner.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/code_diff_jax_pilot.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/code_scope_grain_pilot.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/compare-wiring.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/compute-alpha-gap.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/compute-alpha-rho.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/concept_authority.py` | Repair, Stage 1 | Keep configured path; reject empty-authority fallback; ship index |
| `scripts/concept_shuttle.py` | Repair/review, Stage 1 | Referenced by scripts/warp_debt_report.py; distinguish checkout, external data and sibling roots |
| `scripts/ct_anatomy_slice.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/ct_fresh_extract.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/ct_nlp_side_by_side.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/curriculum_propose.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/daily_reembed.sh` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/demo-discover-terms.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/diffsub_emit.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/diffsub_emit_stub.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/diffsub_scope_dump.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/dp_anatomy_html.py` | Repair/review, Stage 1 | Referenced by scripts/rr_compositor.py; distinguish checkout, external data and sibling roots |
| `scripts/dp_batch.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/dp_capabilities/binders.py` | Repair/review, Stage 1 | Referenced by scripts/dp_paper_view.py; distinguish checkout, external data and sibling roots |
| `scripts/dp_paper_view.py` | Repair/review, Stage 1 | Referenced by scripts/dp_anatomy_html.py; distinguish checkout, external data and sibling roots |
| `scripts/drainage_flow.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/dry_basin_miner.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/explore-p4-path2-A-amgm.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/explore-p4-path2-A-structure.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/explore-p4-path2-A-taylor.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/explore-p4-path2-codex-cycle3.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/explore-p4-path2-codex-handoff2.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/explore-p4-path2-codex.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/explore-p4-path2-gap-analysis.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/explore-p4-path2-gap-analysis2.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/extract_se_threads.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/fold_embed/gfn_gold_loader.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/fold_embed/gfn_seed_v0.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/fold_embed/mk_dataset.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/futon6-status.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/iatc_alignment_passA.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/iatc_mose_scan.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/linode_stepper.py` | Keep/adapt, Stage 1 | Interpreter override; preserve correct argument quoting |
| `scripts/log_loss.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/magnet_probe_extract.bb` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/magnet_quality_probe.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mark3_coverage_model.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mark3_expository_loop.py` | Repair, Stage 3 | Cap requires selected/deferred accounting |
| `scripts/mark4_apm_random_scope_disagreement.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mark4_proof_keyword_retrieval.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mark4_proofs_to_tex.sh` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/marks_to_hx.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/meme_consume.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/meme_mine_joint.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/meme_target_sample.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mine-question-patterns.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mine_coq_defs.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mine_mathlib_defs.py` | Repair/review, Stage 1 | Referenced by scripts/mine_prose_def.py; distinguish checkout, external data and sibling roots |
| `scripts/mine_prose_def.py` | Repair/review, Stage 1 | Referenced by scripts/concept_shuttle.py; distinguish checkout, external data and sibling roots |
| `scripts/mission_anatomy_profile.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mission_carpet.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mission_carpet_variants.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mission_concept_tag.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mission_domain_classify.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mission_dossier.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mission_efe_field.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mission_efe_scope_dump.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mission_embed_diagnostics.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mission_fold.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mission_landscape_mandala.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mission_mine_moves.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mission_pattern_scopes.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mission_phylo_mandala.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mission_phylogeny.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mission_scope_bindings.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mission_scope_detect.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mission_structure_embed.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/mission_triple_miner.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/paleo_topography.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/paper_title_card.py` | Repair/review, Stage 1 | Referenced by scripts/build_golden_paper.py; distinguish checkout, external data and sibling roots |
| `scripts/pattern_phylogeny.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/postfilter_meme_mine_leak.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/preflight.py` | Repair/review, Stage 1 | Referenced by entry point; distinguish checkout, external data and sibling roots |
| `scripts/prelim_anatomy_atlas.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/process-all-planetmath.sh` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/promote_to_proof_layer.bb` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/proof_scope_audit.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/proof_tex_audit.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/proofread_loop.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/publish-efe-field.sh` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/reflow.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/reflow_campaign.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/refresh_pattern_attestation.sh` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/render_anatomy_pdf.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/ricci_bottleneck_pilot.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/rr_layer_weft.py` | Repair/review, Stage 1 | Referenced by scripts/render_run.py; distinguish checkout, external data and sibling roots |
| `scripts/run-frontier-superpod-trial.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/selection_harness.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/session_mission_comb.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/session_scope_view.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/session_thread_plot.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/session_threads.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/setup-ct-run.sh` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/sfc_concept_aggregate.py` | Repair/review, Stage 1 | Referenced by scripts/warp_run.py; distinguish checkout, external data and sibling roots |
| `scripts/starmap_to_capability_graph.bb` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/thread_orbit.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/thread_orbits.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/validate-ct.py` | Repair/review, Stage 1 | Referenced by scripts/nlab-wiring.py; distinguish checkout, external data and sibling roots |
| `scripts/verify-blocking-results.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/verify_check.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `scripts/warp_bib.py` | Repair/review, Stage 1 | Referenced by scripts/warp_run.py; distinguish checkout, external data and sibling roots |
| `scripts/warp_concept_embed.py` | Repair/review, Stage 1 | Referenced by scripts/warp_run.py; distinguish checkout, external data and sibling roots |
| `scripts/warp_concept_graph.py` | Repair/review, Stage 1 | Referenced by scripts/warp_run.py; distinguish checkout, external data and sibling roots |
| `scripts/warp_concept_usage.py` | Repair/review, Stage 1 | Referenced by scripts/warp_run.py; distinguish checkout, external data and sibling roots |
| `scripts/warp_concordance.py` | Repair/review, Stage 1 | Referenced by scripts/warp_run.py; distinguish checkout, external data and sibling roots |
| `scripts/warp_debt_report.py` | Repair/review, Stage 1 | Referenced by scripts/warp_run.py; distinguish checkout, external data and sibling roots |
| `scripts/warp_def_snippets.py` | Repair/review, Stage 1 | Referenced by scripts/warp_run.py; distinguish checkout, external data and sibling roots |
| `scripts/warp_defined_pass.py` | Repair/review, Stage 1 | Referenced by scripts/warp_run.py; distinguish checkout, external data and sibling roots |
| `scripts/warp_greatest_hits.py` | Repair/review, Stage 1 | Referenced by scripts/warp_run.py; distinguish checkout, external data and sibling roots |
| `scripts/warp_hitlist.py` | Repair/review, Stage 1 | Referenced by scripts/warp_run.py; distinguish checkout, external data and sibling roots |
| `scripts/warp_or_curvature.py` | Repair/review, Stage 1 | Referenced by scripts/warp_run.py; distinguish checkout, external data and sibling roots |
| `scripts/warp_paper_landscape.py` | Repair/review, Stage 1 | Referenced by scripts/warp_run.py; distinguish checkout, external data and sibling roots |
| `scripts/warp_run.py` | Repair/review, Stage 1 | Referenced by scripts/warp_substrate_check.py; distinguish checkout, external data and sibling roots |
| `scripts/warp_salingaros.py` | Repair/review, Stage 1 | Referenced by scripts/warp_run.py; distinguish checkout, external data and sibling roots |
| `src/futon6/peradam_cert.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `tests/test_futon6_status.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
| `tests/test_mark2.py` | Defer, Stage 5 | Not reached by this inventory; separate workflow review needed |
