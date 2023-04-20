#my_transfer_method = TransferSimpleHDF5(
#    debug=args.debug, bulk_storage_path=Path(args.save_result_dir)
#)
if args.application == "nt3":
    problem_instance = NT3Problem(
        train_path=args.train_data_path,
        test_path=args.test_data_path,
        num_epochs=args.num_epochs,
        problem_size=args.candle_problem_size,
    )
    problem = problem_instance.setup_problem()
    # We want to load dataset and put it in Ray's global store  before the search process starts
    if args.load_data_mode in ["ray", "mpicomm"]:
        train_data = global_load_data.load_preproc_nt3_data_from_file(
            args.train_data_path, args.test_data_path, 2
        )
    # problem_instance.test_problem()

elif args.application == "synthetic":
    from synthetic_problem import SyntheticProblem

    problem_instance = SyntheticProblem(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        space=args.synthetic_space,
    )
    problem = problem_instance.setup_problem()
    # Pre-load the data from disk
    if args.load_data_mode in ["ray", "mpicomm"]:
        train_data = load_data()

elif args.application == "uno":
    problem_instance = UnoProblem(
        batch_size=args.batch_size, num_epochs=args.num_epochs
    )
    problem = problem_instance.setup_problem(
        batch_size=args.batch_size, num_epochs=args.num_epochs
    )
    if args.load_data_mode in ["ray", "mpicomm"]:
        if args.train_data_path is not None:
            train_data = global_load_data.load_uno_data_from_file(args.train_data_path)
        else:
            train_data = global_load_data.load_uno_data_fake()

elif args.application == "mnist":
    problem_instance = mnist.MNISTProblem(
        batch_size=args.batch_size, num_epochs=args.num_epochs
    )
    problem = problem_instance.setup_problem(
        batch_size=args.batch_size, num_epochs=args.num_epochs
    )
    if args.load_data_mode in ["ray", "mpicomm"]:
        train_data = mnist.load_data2()

elif args.application == "attn":
    problem_instance = AttnProblem(num_epochs=args.num_epochs, rel_size=args.attn_problem_size)
    problem = problem_instance.setup_problem()
    train_data = global_load_data.attn_load_data()
else:
    problem_instance = ComboProblem(num_epochs=args.num_epochs)
    problem = problem_instance.setup_problem()
    train_data = global_load_data.combo_load_data()

# Put the dataset in ray's global KV store
dat_id = None
if args.load_data_mode == "ray":
    if ray.is_initialized() == False:
        print(args.ray_head)
        ray.init(args.ray_head)
    dat_id = ray.put(train_data)
elif args.load_data_mode == "mpicomm":
    # in this mode, each worker can access the train_data global
    train_data_global = train_data
