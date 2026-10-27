# Local semantic task annotations

`configs/local_tasks.yaml` contains 36 visually reviewed local navigation episodes: 12 in each of three GLB scenes, with four `visible`, four `turn`, and four `disambiguate` episodes per scene. Skokloster castle is the development scene; Van Gogh's room and the apartment are held out. The three scene families describe the actual interiors: a historic palace salon, a stylized artist's bedroom, and a scanned modern apartment.

These are **36 task variants over 12 distinct landmark regions**, four regions per scene. Visible and turn variants deliberately reuse a destination and usually a start position while changing heading; disambiguation variants use contrasting descriptions and another start. They are correlated examples, not 36 independent destinations or a standardized R2R/RxR benchmark. Keep all variants from one scene in the same split. Use `target_region_id` to prevent an instruction shuffle from selecting another wording for the same destination.

## What was checked

On 2026-10-26 the authoring session rendered and visually inspected the three scene surveys, candidate landmark goal views, and all 36 final start/goal pairs through the image viewer. Long routes were shortened along actual navmesh paths, and the changed panels were rendered and inspected again. Two bedroom turn starts that faced missing geometry were changed to interior views and inspected again. `annotation_verified: true` records that review; it does not claim an independent human annotation study.

All 36 frozen starts and goals are navigable on the matching recomputed navmesh, all 36 start-goal paths exist, and every start is strictly outside its goal's success radius. Final shortest paths range from **0.536 to 4.761 m**. Goal regions stay on the room floor; the palace roof and bedroom bed-top surfaces that appeared in unconstrained random sampling were rejected.

The official Habitat-Sim `GreedyGeodesicFollower` also planned and **executed** each route with a 0.25 m forward action and 15 degree turns. All 36 ended within the goal radius in **2–29 decisions including STOP**, below the 60-decision pilot budget. This is an offline geometry and action-budget check using known coordinates. It measures neither language understanding nor perception, and its coordinates/actions are not policy inputs. `budget_feasibility_checked` and `oracle_action_count` preserve this evidence in each frozen row.

## Landmark regions

| Scene / split | Landmark region IDs | Success radius |
| --- | --- | --- |
| `skokloster-castle` / dev | `tiled-stove`, `horse-paintings`, `portrait-and-vase`, `floral-door` | 0.60 m |
| `van-gogh-room` / holdout | `door-by-bed`, `chair-by-cloth`, `chair-by-window`, `bed-footboard` | 0.30 m |
| `apartment_1` / holdout | `round-mirror`, `dining-door`, `end-wall-painting`, `sofa-window` | 0.45 m |

The full `target_region_id` is `<scene_id>:<landmark_region>`. It is stable across visible, turn, and disambiguation variants. A target describes a nearby walkable floor region, rather than an object mesh center. The door instructions ask the agent to approach a door; they do not require opening it. The painting and mirror instructions ask the agent to approach the floor region near the named display.

`disambiguate` episodes use visually inspected attributes and relations: chairs below horse paintings versus chairs by windows; a vase beneath a portrait versus horse paintings; one floral door beside the stove versus another beside chairs and a cabinet; a chair by the green cloth versus a chair by the window table; the bed's footboard versus its pillow end; a dining door versus an end-wall door; a round mirror versus a framed picture; and sofa windows versus dining windows. The instructions make no screen-relative left/right claims. Contrasted referents may require looking around and are not required to appear together in the first image.

The bedroom has very little floor area and four nearby landmark regions. Several tasks need only one or a few forward moves after orienting. Its two chairs, two blue doors, and bed endpoints support useful attribute contrasts, but this scene cannot establish broad room-scale navigation ability. Some source mesh boundaries contain missing geometry; task starts were chosen to face the interior. Scene appearance and floor coordinates remain scene-specific.

## Geometry and rendering contract

The annotation helper uses native official Habitat-Sim, RGB and depth sensors at 256 × 256, 90 degree horizontal field of view, sensor height 1.25 m, agent radius 0.20 m, and agent height 1.50 m. It recomputes the navmesh from the GLB using those dimensions. Match these settings when evaluating the suite; the supplied asset navmeshes may have different clearance settings.

Positions use Habitat world coordinates in metres. Quaternions use `[x, y, z, w]`. `goal_rotation` only points the annotation preview camera toward the landmark and is not a navigation success condition. `annotation_evidence`, `annotation_note`, `annotation_review_date`, and `oracle_action_count` are authoring metadata, not policy inputs. Habitat reports absent machine-readable semantic annotations for these sample GLBs; the landmark labels here came from rendered RGB inspection, not simulator object labels.

## Reproduce and correct annotations

From the repository root, using the native environment:

```bash
.habitat/bin/python3.9 scripts/annotate_tasks.py survey
.habitat/bin/python3.9 scripts/annotate_tasks.py preview --config configs/local_tasks.yaml
```

`survey` saves floor-filtered candidate camera poses to `runs/annotation-preview/survey.json` and one montage per scene. It does not change the task suite. Its explicit floor-height limits reject known roof/furniture surfaces for these three fixtures.

`preview` renders every frozen start and goal into three `*-tasks.jpg` montages and writes `runs/annotation-preview/validation.json`. It checks navigability, path existence, the strict outside-radius start condition, and the executed discrete oracle route. `oracle_actions` includes the terminal STOP. The helper never sets `annotation_verified` automatically. It exits with an error if geometric validation fails; inspect the budget fields as well. Generated previews and validation records are ignored under `runs/` and can be rebuilt locally.

To correct a row, first set `annotation_verified: false`, edit its instruction or poses, and rerun `preview`. Inspect its initial and goal panels, verify that the named destination is recognizable and distinct from any contrasted destination, and check the path and action-budget fields before restoring the flag. If changing a shared region, update every variant with the same `target_region_id`, then review all affected panels. Keep the success radius small enough that neighboring regions remain distinct, and ensure starts do not already satisfy success.

No remote model API was used to create or validate these annotations. The suite is a local semantic navigation pilot; its verification does not constitute a model performance result.
