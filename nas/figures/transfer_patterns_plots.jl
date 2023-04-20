#!/usr/bin/env julia
#installation
using Pkg
pkgs = [ "Plots", "StatsPlots", "Pandas", "DataFrames", "Glob", "CSV", "JSON", "StatsBase", "Printf"]
Pkg.add(pkgs)

# load installed packages
using Plots: plot
using StatsPlots
using Pandas: read_pickle, concat
using DataFrames
using Glob
using CSV
using JSON
using StatsBase
using Printf
################################################################################
#Configuration Variables
################################################################################
outputdir=expanduser("~/Downloads")
datadir=expanduser("~/scratch/deephyper")
names = Dict("attn_64gpu_notransfer"=>"DH-No Transfer 64", "attn_128gpu_notransfer"=>"DH-No Transfer 128", "attn_256gpu_notransfer"=>"DH-No Transfer 256", "largeattn_ds_64gpu"=>"EvoStore 64", "largeattn_ds_256gpu"=>"EvoStore 256", "largeattn_ds_128gpu"=>"EvoStore 128", "largeattn_thread_colocated_64gpu_ds_rerun" => "EvoStore 64")
gpus = Dict("attn_64gpu_notransfer"=>64,"attn_128gpu_notransfer"=>128, "attn_256gpu_notransfer"=>256, "largeattn_ds_64gpu"=>64, "largeattn_ds_128gpu"=>128, "largeattn_ds_256gpu"=>256, "largeattn_thread_colocated_64gpu_ds_rerun"=>64)
method = Dict("attn_64gpu_notransfer"=>"DH-No Transfer","attn_128gpu_notransfer"=>"DH-No Transfer", "attn_256gpu_notransfer"=>"No Transfer", "largeattn_ds_64gpu"=>"EvoStore", "largeattn_ds_128gpu"=>"EvoStore", "largeattn_ds_256gpu"=>"EvoStore", "largeattn_thread_colocated_64gpu_ds_rerun"=>"EvoStore")
################################################################################
# Helper functions
################################################################################
sequences(seqs; top_k=1, k_grams=1) = last(sort(collect(countmap(map(x->x[1:k_grams], seqs))), by=last), top_k)
flip(t) = (t[2], t[1])
function to_time_variation(time_seq, value_seq; parts=10, boundries=missing)
    n = length(time_seq)
    df = DataFrame(value=value_seq, time=time_seq)
    results = DataFrame(mean=Float64[], max=Float64[], std=Float64[], start_time=Float64[], end_time=Float64[], width=Float64[], count=UInt64[])
    if ismissing(boundries)
        lower, upper = extrema(time_seq)
        width = (upper-lower)/parts
        loopbounds = collect(lower:(upper-lower)/(parts):upper)[begin:end-1]
    else
        lower, upper = extrema(time_seq)
        width = boundries[2] - boundries[1]
        loopbounds = boundries
    end
    for start in loopbounds
        p = filter(:time => x -> (start <= x <= start+width), df)
        push!(results, (mean(p.value), maximum(p.value; init=-floatmax(Float64)), std(p.value), start, start+width, width, size(p,1)))
    end
    results
end
to_time_variation(value_seq; kwargs...) = to_time_variation(1:length(value_seq), value_seq; kwargs...)
function kgrams(seqs; top_k=30, k_grams=3, basename="")
    s = Set()
    for i=1:length(seqs)
        top_counts = sequences(seqs[1:i]; k_grams=k_grams, top_k=top_k)
        for pattern in first.(top_counts)
            push!(s, pattern)
        end
    end
    x_pos = Dict(flip.(enumerate(s)))
    x_max = maximum(values(x_pos))
    anim = @animate for i=1:length(seqs)
        top_counts = sequences(seqs[1:i]; k_grams=k_grams, top_k=top_k)
        p = plot()
        positions = []
        for p in first.(top_counts)
            push!(positions, x_pos[p])
        end
        bar!(p, positions, last.(top_counts), xlim=(1, x_max), title="iter =$i, $basename-top_k=$top_k-k_grams=$k_grams")
    end
    gif(anim, "$basename-top_k=$top_k-k_grams=$k_grams.gif", fps=15)
end
function sliding_kgrams(seqs; top_k=30, k_grams=3, window_size=100, basename="")
    s = Set()
    for i=1:length(seqs) - window_size
        top_counts = sequences(seqs[i:i+window_size]; k_grams=k_grams, top_k=top_k)
        for pattern in first.(top_counts)
            push!(s, pattern)
        end
    end
    x_pos = Dict(flip.(enumerate(s)))
    x_max = maximum(values(x_pos))
    anim = @animate for i=1:length(seqs) - window_size
        top_counts = sequences(seqs[i:i+window_size]; k_grams=k_grams, top_k=top_k)
        p = plot()
        positions = []
        for p in first.(top_counts)
            push!(positions, x_pos[p])
        end
        bar!(p, positions, last.(top_counts), xlim=(1, x_max), title="iter =$i")
    end
    gif(anim, "$basename-top_k=$top_k-k_grams=$k_grams-sliding.gif", fps=15)
end
pandas_to_julia(pd_df) = DataFrames.DataFrame([col => collect(Any, pd_df[col]) for col in pd_df.pyo.columns])
struct Traces
    entries
    transfers
    stores
end
function read_pickle_add_file(x)
    df = read_pickle(x)
    df["filename"] = parse(Int,match(r"(\d+)", basename(x), 1).captures[1])
    return df
end
function read_traces(dir =".")
    entries = glob("*traces.pkl", dir) .|> read_pickle_add_file |> concat |> pandas_to_julia;
    rename!(entries, Dict(:return_candidates => "metadata"));
    entries.accuracy = map(x -> x["accuracy"], entries.metadata);
    entries.tensorsizes = map(x -> x["tensor_sizes"], entries.metadata);
    entries.tensorids = map(x -> x["tensor_ids"], entries.metadata);
    entries.query = map(x -> x["query"], entries.metadata);
    entries.timestamp = entries.timestamp .- minimum(entries.timestamp);
    entries.totalsize = map(x -> sum(x; init=0), entries.tensorsizes);
    entries.totallayers = map(length, entries.tensorsizes);
    entries.dir .= basename(dir)
    sort!(entries, :timestamp);
    entries.maxacc = accumulate(max, entries.accuracy);
    transfers = filter(:action => ==("transfer"), entries);
    stores = filter(:action => ==("store"), entries);
    Traces(entries, transfers, stores)
end
struct Results
    results
    arch_seqs
end
function read_results(dir)
    results = CSV.read(joinpath(dir, "results.csv"), DataFrame);
    sort!(results, :timestamp_gather);
    results.maxobjective = accumulate(max, results.objective);
    results.dir .= basename(dir)
    results.times = (results.timestamp_gather - results.timestamp_submit);
    arch_seqs = map(l -> parse.(Int, split(l[2:end-1], ',')), results.arch_seq);
    Results(results, arch_seqs)
end
function parse_and_identify(filename)
    j = JSON.parse(open(filename))
    j["job_id"] = parse(Int, match(r"(\d+)", basename(filename), 1).captures[1])
    j
end
struct History
    history
    fullhistory
end
function read_history(dir)
    results = read_results(dir)
    history = glob("save/history/*.json", dir) .|> parse_and_identify |> DataFrame;
    fullhistory = innerjoin(history, results.results, on=:job_id);
    fullhistory.acc = first.(fullhistory.acc)
    fullhistory
end
function times_to_objectives(data, objective_property, time_property; thresholds=range(.60, 1.0, length=20), properties=[])
    time_to_objective = DataFrame(timestamp=Union{Missing, Float64}[], objective=Union{Float64,Missing}[], threshold=Float64[])
    for p in properties
        time_to_objective[!, p] = missings(eltype(getproperty(data,p)), nrow(time_to_objective))
    end
    for threshold = thresholds
        f = filter(objective_property => x -> (x >threshold) , sort!(data, time_property))
        #get defaults from the dataset
        props = Dict{String, Any}()
        for p in properties
            props[String(p)]=getproperty(first(data),p)
        end
        if ! isempty(f)
            entry = first(f)
            ts = getproperty(entry, time_property)
            obj = getproperty(entry, objective_property)
            for p in properties
                props[String(p)]= getproperty(entry,p)
            end
        else
            obj = missing
            ts = missing
        end
        t = threshold
        push!(time_to_objective, (ts, obj, t, values(props)...))
    end
    time_to_objective
end

#################################################################################
#what is in the entries pkl files
#################################################################################

# note we want :acc, not :accuracy here which is the :val_acc field
# until the datasets are updated, these plots won't make sense

traces = read_traces(expanduser("$datadir/attn_20nodes"))

p = plot()
@df traces.transfers scatter!(p, :timestamp, :accuracy, title="Datastates: ATTN", ms=1.5, ma=0.5, group=:action, legend_column=-1, ylim=(.8,.97), xlabel="Timestamp (sec)", ylabel="Validation Accuracy")

savefig(p, "$outputdir/accuracy_scatter.pdf")


# plot of accuracy vs cummulative accuracy for stores
p = plot()
@df traces.stores scatter!(p, :timestamp, :accuracy, title="Datastates: ATTN", label="Validation Accuracy", ms=1.5, ma=0.5, ylim=(.96, .97), legend_column=-1, xlabel="Timestamp (sec)", ylabel="Validation Accuracy")
@df traces.stores plot!(p, :timestamp, :maxacc, label="Cummulate Max Validation Accuracy", legend=:outerbottom)

savefig(p, "$outputdir/stores_cumacc.pdf")

# plot of accuracy vs cummulative accuracy for transfers
p = plot()
@df traces.transfers scatter!(p, :timestamp, :accuracy, title="Datastates: ATTN", label="Validation Accuracy", ms=1.5, ma=0.5, ylim=(.96, .97), legend_column=-1, xlabel="Timestamp (sec)", ylabel="Validation Accuracy")
@df traces.transfers plot!(p, :timestamp, :maxacc, label="Cummulate Max Validation Accuracy", legend=:outerbottom)

savefig(p, "$outputdir/transfers_cumacc.pdf")

# plot of accuracy vs cummulative accuracy for entries
p = plot()
@df traces.entries scatter!(p, :timestamp, :accuracy, title="Datastates: ATTN", label="Validation Accuracy", ms=1.5, ma=0.5, ylim=(.96, .97), legend_column=-1, xlabel="Timestamp (sec)", ylabel="Validation Accuracy")
@df traces.entries plot!(p, :timestamp, :maxacc, label="Cummulate Max Validation Accuracy", legend=:outerbottom)

savefig(p, "$outputdir/entries_cumacc.pdf")

#plot of size over time
p = plot()
@df to_time_variation(traces.transfers.timestamp, traces.transfers.totalsize, parts=100) plot!(p, :start_time, :mean, yerr=:std, barwidth=:width, xlabel="search time (sec)", ylabel="totalsize (bytes)")

savefig(p, "$outputdir/entries_sizeovertime.pdf")

#model size over time
p = plot()
@df to_time_variation(traces.transfers.timestamp, traces.transfers.totallayers, parts=100) plot!(p, :start_time, :mean, yerr=:std, barwidth=:width, xlabel="search time (sec)", ylabel="totalsize (layers)")

savefig(p, "$outputdir/entries_layersovertime.pdf")


#################################################################################
# what's in the results file?
#################################################################################

# note we want :acc, not :accuracy here which is the :val_acc field
# until the datasets are updated, these plots won't make sense
#
results = read_results(expanduser("$datadir/attn_20nodes/"))

p = plot()
@df results.results scatter!(p, :timestamp_gather, :objective, title="Datastates: ATTN", label="Validation Accuracy", ms=1.5, ma=0.5, legend_column=-1, ylim=(.95, 1.01), xlim=(0,500), xlabel="Timestamp (sec)", ylabel="Validation Accuracy")
@df results.results plot!(p, :timestamp_gather, :maxobjective, label="Cummulate Max Validation Accuracy", legend=:outerbottom)

savefig(p, "$outputdir/results_cumacc.pdf")

#kgrams
kgrams(results.arch_seqs, top_k=30, k_grams=3, basename="$outputdir/attn")
sliding_kgrams(results.arch_seqs, top_k=30, k_grams=3, basename="$outputdir/attn")

#histogram over time from beginning to now
anim = @animate for i=1:length(results.arch_seqs)
    p = plot()
    histogram!(p, (results.results.timestamp_gather - results.results.timestamp_submit)[1:i], title="iter = $i")
end
gif(anim, "$outputdir/times_since_beginning.gif", fps=15)

#histogram over time with a sliding window
window_size=100
times = (results.results.timestamp_gather - results.results.timestamp_submit)
anim = @animate for i=1:length(results.results.arch_seq)-window_size
    p = plot()
    histogram!(p, times[i:i+window_size], title="iter = $i")
end
gif(anim, "$outputdir/times_sliding.gif", fps=15)

#search time over time
p = plot()
@df to_time_variation(results.results.timestamp_gather, results.results.times, parts=100) plot!(p, :start_time, :mean, yerr=:std, barwidth=:width, xlabel="search time (sec)", ylabel="task time (sec)")

savefig(p, "$outputdir/searchtime_over_time.pdf")
 

#accuracy over time
p = plot()
@df to_time_variation(results.results.timestamp_gather, results.results.objective, parts=100) plot!(p, :start_time, :mean, yerr=:std, barwidth=:width, xlabel="search time (sec)", ylabel="validation accuracy")

savefig(p, "$outputdir/accuracy_over_time.pdf")


################################################################################
# splunking in the history JSON files
################################################################################
fullhistory = read_history(expanduser("$datadir/attn_20nodes"))

#breakout transfer, training, and storing time, layers transfered, size
p = plot()
@df to_time_variation(fullhistory.timestamp_gather, fullhistory.training_time, parts=100) plot!(p, :start_time, :mean, yerr=:std, barwidth=:width, xlabel="search time (sec)", ylabel="training time (sec)")

p = plot()
@df to_time_variation(fullhistory.timestamp_gather, fullhistory.transfer_time, parts=100) plot!(p, :start_time, :mean, yerr=:std, barwidth=:width, xlabel="search time (sec)", ylabel="transfer time (sec)")

p = plot()
@df to_time_variation(fullhistory.timestamp_gather, fullhistory.storing_time, parts=100) plot!(p, :start_time, :mean, yerr=:std, barwidth=:width, xlabel="search time (sec)", ylabel="storing time (sec)")

p = plot()
@df to_time_variation(fullhistory.timestamp_gather, fullhistory.num_layers_transferred, parts=100) plot!(p, :start_time, :mean, yerr=:std, barwidth=:width, xlabel="search time (sec)", ylabel="# of layers transfered")

p = plot()
@df to_time_variation(fullhistory.timestamp_gather, fullhistory.n_parameters, parts=100) plot!(p, :start_time, :mean, yerr=:std, barwidth=:width, xlabel="search time (sec)", ylabel="n_paramaters")

# combined plot
p = plot()
@df to_time_variation(fullhistory.timestamp_gather, fullhistory.training_time, parts=100) plot!(p, :start_time, :max, barwidth=:width, xlabel="search time (sec)", ylabel="units (sec, million parameters, or layers)", label="training time")
@df to_time_variation(fullhistory.timestamp_gather, fullhistory.transfer_time, parts=100) plot!(p, :start_time, :max, barwidth=:width, xlabel="search time (sec)", label="transfer time")
@df to_time_variation(fullhistory.timestamp_gather, fullhistory.storing_time, parts=100) plot!(p, :start_time, :max, barwidth=:width, xlabel="search time (sec)", label="storing time")
@df to_time_variation(fullhistory.timestamp_gather, fullhistory.n_parameters ./ 10^6, parts=100) plot!(p, :start_time, :mean, barwidth=:width, xlabel="search time (sec)", label="million paramaters")
@df to_time_variation(fullhistory.timestamp_gather, fullhistory.num_layers_transferred, parts=100) plot!(p, :start_time, :mean, barwidth=:width, xlabel="search time (sec)", label="num layers transfered")

savefig(p, "$outputdir/combined_breakdown.png")

################################################################################
# Let's disprove that caching the dataset and startup drive the early search-time
################################################################################

sort!(traces.stores, :timestamp);
g = groupby(traces.stores, :filename);
f = combine(g, :timestamp => (x -> x[2]-x[1]) => :first_task);
s = combine(g, :timestamp => (x -> x[3]-x[2]) => :second_task);
j = innerjoin(f,s, on=:filename);
p = plot()
histogram!(p, j.second_task - j.first_task, title="first vs second task time per worker", xlabel="time (sec)", ylabel="frequency",legend=false)

savefig(p, "$outputdir/first_second_task_duration.png")


################################################################################
# Let's compute the time to the .95, .99, .995
################################################################################
times_to_objectives(fullhistory, :acc, :timestamp_gather; thresholds=.6:.01:.85, properties=[:dir])

################################################################################
# Let's make comparitive plots from multiple directories
################################################################################

ds_phase1 = read_history("$datadir/attn_32nodes_ds_phase1");
ds_8nodes = read_history("$datadir/attn_8nodes_test_save_model_weights");
ds_phase1_var = to_time_variation(ds_phase1.timestamp_gather, ds_phase1.training_time, parts=10);
ds_phase1_var.dir .= "ds_phase1_var";
ds_8nodes_var = to_time_variation(ds_8nodes.timestamp_gather, ds_8nodes.training_time; boundries = ds_phase1_var.start_time);
ds_8nodes_var.dir .= "ds_8nodes_var";
df = vcat(ds_phase1_var, ds_8nodes_var)
p = plot()
@df df groupedbar!(:start_time, :mean, group=:dir, title="", ylabel="task time (sec)", xlabel="search progress (sec)", legend=:bottomright)


ds_phase1 = read_history(expanduser("$datadir/attn_32nodes_ds_phase1"));
ds_8nodes = read_history(expanduser("$datadir/attn_32nodes_notransfer"));
a = times_to_objectives(ds_phase1, :acc, :timestamp_gather, thresholds=.6:.01:.99, properties=[:dir])
b = times_to_objectives(ds_8nodes, :acc, :timestamp_gather, thresholds=.6:.01:.99, properties=[:dir])
df = vcat(a,b)
p = plot()
@df df plot!(:timestamp, :objective, group=:dir, ylabel="model accuracy", xlabel="search progress (sec)")

files = ["end_to_end_attn/attn_128gpu_notransfer", "end_to_end_attn/largeattn_ds_128gpu"]
df = vcat((times_to_objectives(read_history("$datadir/$f"), :acc, :timestamp_gather, thresholds=.6:.01:.99, properties=[:dir]) for f in files)...)
p = plot()
@df df scatter!(:timestamp, :objective, group=:dir, legend=:bottomright, ylabel="model accuracy", xlabel="search progress (sec)")

# time to quality plot
files = ["end_to_end_attn/attn_128gpu_notransfer","end_to_end_attn/attn_256gpu_notransfer", "end_to_end_attn/largeattn_ds_128gpu", "end_to_end_attn/largeattn_ds_256gpu"]
df = vcat((times_to_objectives(read_history("$datadir/$f"), :acc, :timestamp_gather, thresholds=[.91, .92, .93, .94, .95, .96], properties=[:dir]) for f in files)...)
df.timestamp = coalesce.(df.timestamp, 0)
df.objective = coalesce.(df.objective, 0)
df.names = replace(basename.(df.dir), names...)
df
p = plot()
@df df groupedbar!(:threshold, :timestamp, group=:names, legend_position=:topright, ylabel="Search Time (sec)", ylim=(0,500), xlabel="Target Accuracy", fillstyle=repeat([:/,:-,:x,:+], inner=6), dpi=300, axisfontsize=14, tickfontsize=14, labelfontsize=14)
annotate!(p, [(i-.003, 50, "*") for i=.92:.01:.96]) # No trans 128
annotate!(p, [(i-.001, 50, "*") for i=.95:.01:.96]) # no trans 256
annotate!(p, [(i+.003, 50, "*") for i=.97:.01:.96], legendfontsize=14) # datastates 256
savefig(p, "$outputdir/time_to_objective.png")
p


files = ["end_to_end_attn/attn_64gpu_notransfer","end_to_end_attn/attn_128gpu_notransfer","end_to_end_attn/attn_256gpu_notransfer", "end_to_end_attn/largeattn_ds_128gpu", "end_to_end_attn/largeattn_ds_256gpu", "end_to_end_attn/largeattn_thread_colocated_64gpu_ds_rerun"]
df = vcat((read_history("$datadir/$f") for f in files)...)
df.names = replace(basename.(df.dir), names...)
df = combine(groupby(df, [:dir]), :transfer_time => mean, :training_time => mean, :storing_time => mean, :timestamp_gather => maximum)
df.gpus = replace(basename.(df.dir), gpus...)
df.names = replace(basename.(df.dir), names...)
df.method = replace(basename.(df.dir), method...)
p = plot()
@df df groupedbar!(p, :gpus, :timestamp_gather_maximum, group=:method, fillstyle=repeat([:+, :x], inner=3), xlabel="GPUs", ylabel="Time (sec)", tickfontsize=14, legendfontsize=14, labelfontsize=14, dpi=300)
savefig(p, "$outputdir/speedup-3counts.png")
p


files = ["end_to_end_attn/attn_256gpu_notransfer", "end_to_end_attn/largeattn_ds_256gpu"]
df = vcat((read_history("$datadir/$f") for f in files)...)
p = plot()
@df df scatter!(:timestamp_gather, :acc, group=:dir, legend=:bottomright, ylabel="model accuracy", xlabel="search progress (sec)", ms=1, ma=.5)
savefig(p, "$outputdir/acc_scatter3.pdf")
p


# scatter plots showing quality
markers = [:x, :+]
for gpus in [64, 128,256]
files = ["end_to_end_attn/attn_$(gpus)gpu_notransfer", "end_to_end_attn/largeattn_ds_$(gpus)gpu"]
df = vcat((read_history(expanduser("$datadir/$f")) for f in files)...)
df.names = replace(basename.(df.dir), names...)
p = plot()
@df df scatter!(:timestamp_gather, :acc, group=:names, legend=:bottomright, ms=3, ylim=(.8,1), ylabel="Model Accuracy", xlabel="Search Progress (sec)", markershape=markers[groupindices(groupby(df, "names"))], tickfontsize=14, legendfontsize=14)
savefig(p, "$outputdir/acc_scatter$gpus.pdf")
p
end

# (bogdan, didn't use) scalability with GPUs: ngpus on x, y=time to solution, group=threshold
files = ["end_to_end_attn/attn_64gpu_notransfer", "end_to_end_attn/largeattn_ds_128gpu", "end_to_end_attn/attn_128gpu_notransfer", "end_to_end_attn/largeattn_ds_128gpu", "end_to_end_attn/attn_256gpu_notransfer", "end_to_end_attn/largeattn_ds_256gpu"]
df = vcat((times_to_objectives(read_history("$datadir/$f"), :acc, :timestamp_gather, thresholds=[.9, .91, .92, .93, .94, .95, .96, .97], properties=[:dir]) for f in files)...)
df.names = replace(basename.(df.dir), names...)
df.gpus = replace(basename.(df.dir), gpus...)
df.method = replace(basename.(df.dir), method...)
df.timestamp = coalesce.(df.timestamp, 0)
for threshold = .90:.01:.97
p = plot()
@df filter(:threshold => x -> x == threshold, df) groupedbar!(p, :gpus, :timestamp, group=:method, xlabel="GPUs", ylabel="time to solution (sec)")
savefig(p, "$outputdir/scalability-$threshold.pdf")
end


files = ["end_to_end_attn/attn_128gpu_notransfer", "end_to_end_attn/largeattn_ds_128gpu"]
h = read_history("$datadir/" * files[1]);
scale = to_time_variation(h.timestamp_gather, h.acc, parts=10);
df = vcat(map(files) do f
    history = read_history("$datadir/$f");
    history_var = to_time_variation(history.timestamp_gather, history.acc, boundries=scale.start_time);
    history_var = filter(:mean => !isnan, history_var)
    history_var.file .= basename(f)
    history_var
end...)
p = plot()
@df df plot!(p, :start_time, :max, group=:file,  ylim=(.7, 1.0), ylabel="accuracy", xlabel="time", legend=:bottom, legendcolumns=2)
savefig(p, "$outputdir/max_and_mean_acc_overtime.pdf")
p

ds_phase1_var.dir .= "ds_phase1_var";
ds_8nodes_var = to_time_variation(ds_8nodes.timestamp_gather, ds_8nodes.acc; boundries = ds_phase1_var.start_time);
ds_8nodes_var.dir .= "ds_8nodes_var";
df = vcat(ds_phase1_var, ds_8nodes_var)
p = plot()
@df df groupedbar!(:start_time, :mean, group=:dir, title="", ylabel="task time (sec)", xlabel="search progress (sec)", legend=:bottomright)



files = ["attn_8nodes_ds_phase1_pop_50_iter_1000", "attn_8nodes_ds_nophase1_pop_50_iter_1000", "attn_8nodes_notransfer_pop_50_iter_1000"]
df = vcat((read_history("$datadir/$f") for f in files)...)
d = transform(groupby(sort(df, :timestamp_gather),:dir) , :acc => (x -> accumulate(max, x)) => :max_acc)
p = plot()
#@df d scatter!(p,:timestamp_gather, :acc, group=:dir, legend=:bottomright, ms=1, ma=.3)
@df d plot!(p,:timestamp_gather, :max_acc, group=:dir)
savefig(p, "$outputdir/cummac_acc.pdf")
p

################################################################################
# Microbenchmarks
################################################################################
function extract_metadata(path)
    _, date, config, operation = splitpath(path)
    pop, clients, servers, models, backend, full = match(r"p([a-z]+-\d+)-t(\d+)-s(\d+)-m(\d+)-([a-z]+)-full-(\d)", s).captures
    op = occursin("query", operation) ? "query" : "store"
    (date=date, operation=op, pop=pop, clients=clients, servers=servers, models=models, backend=backend, full=full, path=path)
end
microbench = mapreduce(x -> joinpath.(x[1],x[3]), vcat, walkdir(".")) .|> extract_metadata |> DataFrame


################################################################################
# Debugging
################################################################################

# are the arch seq's the same?
s1 = Set(first(read_results("$datadir/attn_8nodes_ds_phase1_pop_50_iter_1000").arch_seqs,50));
s2 = Set(first(read_results("$datadir/attn_8nodes_ds_nophase1_pop_50_iter_1000").arch_seqs, 50));
s3 = Set(first(read_results("$datadir/attn_8nodes_notransfer_pop_50_iter_1000").arch_seqs,50));

s1 = Set(first(read_results("$datadir/attn_8nodes_ds_phase1_pop_50_iter_1000").results.objective,1));
s2 = Set(first(read_results("$datadir/attn_8nodes_ds_nophase1_pop_50_iter_1000").results.objective, 1));
s3 = Set(first(read_results("$datadir/attn_8nodes_notransfer_pop_50_iter_1000").results.objective,1));



################################################################################
# Metadata performance
################################################################################

function extract_metadata(path)
    try
    _, config, operation  = splitpath(path)[end-2:end]
        op = occursin("query", operation) ? "query" : "store"
        method, readers, population, matches, models = match(r"([a-z]+)-(\d+)r-([a-z]+-\d+)-(\d+)m-(\d+)k", config).captures
        f = CSV.read(path, DataFrame)
        data_mean = mean(f[!, "0"])
        data_std = std(f[!, "0"])
        data_max = maximum(f[!, "0"])
        [(method=method, readers=parse(Int,readers), population=population, matches=parse(Int,matches), models=parse(Int,models), mean=data_mean, std=data_std, max=data_max, operation=op)]
    catch e
        println(config)
        []
    end
end
function extract_metadata_ds(path)
    try
        _, config, operation = splitpath(path)[end-2:end]
        op = occursin("query", operation) ? "query" : "store"
        population, readers, servers, models, method, full  = match(r"p([a-z]+-\d+)-t(\d+)-s(\d+)-m(\d+)-([a-z]+)-full-(\d)", config).captures
        matches="0"
        f = CSV.read(path, DataFrame)
        data_mean = mean(f[!, "0"])
        data_std = std(f[!, "0"])
        data_max = maximum(f[!, "0"])
        [(method=method, readers=parse(Int,readers), population=population, matches=parse(Int,matches), models=parse(Int,models), mean=data_mean, std=data_std, max=data_max, operation=op)]
    catch e
        println(e)
    end
end
redis = vcat((filter(x -> endswith(x, ".csv"), mapreduce(x -> joinpath.(x[1],x[3]), vcat, walkdir("$datadir/microbenchmarks/MockModelBench"))) .|> extract_metadata)...) |> DataFrame
ds = vcat((filter(x -> endswith(x, ".csv"), mapreduce(x -> joinpath.(x[1],x[3]), vcat, walkdir("$datadir/microbenchmarks/2023-04-04"))) .|> extract_metadata_ds)...) |> DataFrame
rf = filter(x -> x.population == "seq-50" && x.operation=="query" && x.models==6 && x.matches==0, redis)
dsf = filter(x -> x.population == "seq-50" && x.operation=="query" && x.models==6 && x.matches==0, ds)
df = vcat(rf,dsf)
df.readers2 = round.(Int,log2.(df.readers))
df.readers2v = 2 .^ df.readers2
df = filter(x -> x.readers2v != 4, df) # filter missing
df = filter(x -> x.readers2v != 2, df) # filter missing
df = filter(x -> x.readers2v != 16, df) # filter missing
df.names = replace(df.method, Dict("redis" => "Redis-Queries", "datastates"=>"EvoStore")...)
df.readers2s = [@sprintf("%3d", i) for i in df.readers2v]
df.method = string.(df.method)

p = plot()
groupedbar!(p, df.readers2s, df.mean, legend=:topright, ylabel="Mean Query Time (sec)", xlabel="GPUs", group=df.names, fillstyle=[:x  :+], dpi=300, tickfontsize=14, labelfontsize=14, legendfontpointsize=20)
annotate!(p, [(i+2.7,3, "*") for i=1:4], legendfontsize=14)
savefig("$outputdir/microbenchmark_query.png")
p

p = plot()
groupedbar!(p, df.readers2s, log10.((1000 ./ df.mean)), legend=:bottomleft, ylabel="Query Bandwidth (queries/sec)", xlabel="GPUs", group=df.names, fillstyle=[:x  :+], dpi=300, tickfontsize=14, labelfontsize=14, yaxis=(formatter=(x -> string(Int(10^x)))))
annotate!(p, [(i+2.7,3, "*") for i=1:4], legendfontsize=14)
savefig("$outputdir/microbenchmark_query_bw.png")
p

################################################################################
# partial writes
################################################################################

# bandwidth version
function extract_partial_writes(path)
    try
        gpus_s, method_s = splitpath(path)[end-1:end]
        data = read_pickle(path)
        data = stack(DataFrame(data), variable_name=:percent, value_name=:time)
        gpus = parse(Int,match(r"(\d+)", gpus_s).captures[1])
        data.gpus .= @sprintf("%3d", gpus)
        method = string(match(r"([a-z0-9]+)_",method_s).captures[1])
        data.method .= method
        data.percent .= parse.(Int, data.percent)
        data.time .= (4 ./ data.time)
        if method == "ds"
            data.group = [@sprintf("EvoStore %3d%%", percent) for percent in data.percent]
        else
            data.group .= "HDF5+PFS 100%"
        end
        data
    catch e
        println(path, e)
        []
    end
end
df = vcat((mapreduce(x -> joinpath.(x[1],x[3]), vcat, walkdir("$datadir/microbenchmarks/partialwrites")) .|> extract_partial_writes)...);
df.names = replace(df.method, "ds"=>"EvoStore", "hdf5"=>"HDF5+PFS");
df = filter(:gpus => x -> x != "  1", df)
p = plot()
@df df groupedbar!(p, :gpus, :time, group=:group, dpi=300, tickfontsize=14, legendfontsize=12, labelfontsize=14, fillstyle=[:x :+ :/ :\ :-], ylabel="Aggregate Bandwidth (GB/sec)", xlabel="GPUs", legend=:topleft)
savefig(p, "$outputdir/microbenchmark_partialwrite_bw.png")
p

# time
function extract_partial_writes(path)
    try
        gpus_s, method_s = splitpath(path)[end-1:end]
        data = read_pickle(path)
        data = stack(DataFrame(data), variable_name=:percent, value_name=:time)
        gpus = parse(Int,match(r"(\d+)", gpus_s).captures[1])
        data.gpus .= @sprintf("%3d", gpus)
        method = string(match(r"([a-z0-9]+)_",method_s).captures[1])
        data.method .= method
        data.percent .= parse.(Int, data.percent)
        data.time .*= gpus
        if method == "ds"
            data.group = [@sprintf("EvoStore %3d%%", percent) for percent in data.percent]
        else
            data.group .= "HDF5 100%"
        end
        data
    catch e
        println(path, e)
        []
    end
end
df = vcat((mapreduce(x -> joinpath.(x[1],x[3]), vcat, walkdir("$datadir/microbenchmarks/partialwrites")) .|> extract_partial_writes)...);
df.names = replace(df.method, "ds"=>"EvoStore", "hdf5"=>"HDF5");
df
p = plot()
@df df groupedbar!(p, :gpus, :time, group=:group, dpi=300, tickfontsize=14, legendfontsize=12, labelfontsize=14, fillstyle=[:x :+ :/ :\ :-], ylabel="Average Time (sec)", xlabel="GPUs", legend=:outerright)
savefig(p, "$outputdir/microbenchmark_partialwrite.png")
p

################################################################################
# Space usage
################################################################################

# get results from space.py script
p = plot()
s=150
bar!(p,["HDF5+PFS", "EvoStore", "Deephyper\nHDF5+PFS", "Deephyper\nEvoStore"], [96.52, 26.92, 9.06, 5.10], xlabel="Method", ylabel="Storage Size (GB)", legend=false, dpi=300, tickfontsize=14, labelfontsize=14, size=(4*s, 3*s), fillstyle=[:x :x :x :x])
savefig(p, "$outputdir/sizes.png")
p
