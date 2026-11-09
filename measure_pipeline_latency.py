                "source_sha256": sha256(r["source"]),
            }
            for r in records
        ])
        metadata["status"] = "warming_up"
        save_json(output / "metadata.json", metadata)

        def process_image(source, destination, branch):
            destination.parent.mkdir(parents=True, exist_ok=True)
            if branch in ("raw", "good"):
                shutil.copy2(source, destination)
                return
            image = processing["read_image"](source)
            if branch == "resize_only":
                processed = processing["resize_only"](image)
            elif branch == "low_light":
                processed = processing["enhance"](five, image)
            elif branch == "turbid":
                processed = processing["enhance"](residual, image)
            else:
                raise ValueError(f"Unknown processing branch: {branch}")
            processing["write_image"](destination, processed)

        def execute(record, configuration, work_root, repetition=0, timed=True):
            source = record["source"]
            # Avoid collisions if different source directories contain identical basenames.
            unique_name = f"{sha256_text(str(source.resolve()))[:12]}_{source.name}"
            destination = work_root / configuration / unique_name
            start_ns = time.perf_counter_ns()
            classifier_ms = 0.0
            route = "all"
            if configuration == "dynamic":
                t0 = time.perf_counter_ns()
                route = classify(source)
                classifier_ms = (time.perf_counter_ns() - t0) / 1e6
                branch = route
            elif configuration == "raw":
                branch = "raw"
            elif configuration == "resize_only":
                branch = "resize_only"
            elif configuration == "fixed_five_layer":
                branch = "low_light"
            elif configuration == "fixed_residual":
                branch = "turbid"
            else:
                raise ValueError(f"Unknown configuration: {configuration}")

            p0 = time.perf_counter_ns()
            process_image(source, destination, branch)
            processing_ms = (time.perf_counter_ns() - p0) / 1e6
            d0 = time.perf_counter_ns()
            detector.predict(source=str(destination), **PREDICT_ARGS)
            detector_call_ms = (time.perf_counter_ns() - d0) / 1e6
            total_ms = (time.perf_counter_ns() - start_ns) / 1e6
            return {
                "configuration": configuration,
                "repetition": repetition,
                "filename": source.name,
                "source": str(source),
                "route": route,
                "saved_route": record["saved_route"],
                "classifier_ms": classifier_ms,
                "processing_ms": processing_ms,
                "detector_call_ms": detector_call_ms,
                "total_ms": total_ms,
            }

        with tempfile.TemporaryDirectory(prefix="pipeline_latency_work_", dir=str(ROOT)) as tmp_name:
            work_root = Path(tmp_name)
            print("Warming up models and processing routes...", flush=True)
            first_record = records[0]
            # Warm each fixed configuration.
            for configuration in ("raw", "resize_only", "fixed_five_layer", "fixed_residual"):
                for _ in range(WARMUP_CALLS):
                    execute(first_record, configuration, work_root, timed=False)
            # Warm the dynamic route on a known image from each saved route, so all branches run.
            for route in ("good", "low_light", "turbid"):
                candidates = [r for r in all_records if r["saved_route"] == route]
                if not candidates:
                    raise ValueError(f"No source image available to warm dynamic route {route}.")
                for _ in range(WARMUP_CALLS):
                    execute(candidates[0], "dynamic", work_root, timed=False)

            metadata["status"] = "running"
            save_json(output / "metadata.json", metadata)
            timing_rows = []
            print(
                f"Starting timed benchmark: {len(records)} images x {REPEATS} repeats x "
                f"{len(CONFIGURATIONS)} configurations...",
                flush=True,
            )
            for repetition in range(1, REPEATS + 1):
                run_rng = random.Random(SEED + repetition)
                image_order = list(records)
                config_order = list(CONFIGURATIONS)
                run_rng.shuffle(image_order)
                run_rng.shuffle(config_order)
                for configuration in config_order:
                    for record in image_order:
                        timing_rows.append(execute(record, configuration, work_root,
                                                   repetition=repetition, timed=True))
                    # Checkpoint results after every configuration in case a later run fails.
                    save_csv(output / "timing_observations_partial.csv", timing_rows)
                    print(
                        f"Completed repetition {repetition}/{REPEATS}, configuration {configuration}; "
                        f"{len(timing_rows)} timed observations saved.",
                        flush=True,
                    )

        save_csv(output / "timing_observations.csv", timing_rows)
        partial_path = output / "timing_observations_partial.csv"
        if partial_path.exists():
            partial_path.unlink()
        summaries = summarize(timing_rows)
        save_csv(output / "summary.csv", summaries)
        metadata["status"] = "completed"
        metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["timed_observations"] = len(timing_rows)
        metadata["summary_file"] = str(output / "summary.csv")
        metadata["timings_file"] = str(output / "timing_observations.csv")
        save_json(output / "metadata.json", metadata)
        print(f"Benchmark completed. Results saved to: {output}", flush=True)

    except Exception as exc:
        if metadata.get("status") not in ("failed_route_verification",):
            metadata["status"] = "failed"
            metadata["error"] = repr(exc)
        metadata["traceback"] = traceback.format_exc()