import JACC
JACC._check_install_backend()
JACC.@init_backend

using ReTest
include("common.jl")
include("JACCBench.jl")
include("JACCTests.jl")

if JACCBench.matches(ARGS)
    popfirst!(ARGS)
    filter = JACCBench.getconf().filter
    if isempty(filter)
        retest(JACCBench; spin = false, stats = true)
    else
        retest(JACCBench, filter; spin = false)
    end
else
    try
        if isempty(ARGS)
            retest(JACCTests)
        else
            retest(JACCTests, ARGS)
        end
    catch
        # retest() throws as soon as a testset fails, so every testset
        # defined after the failing one silently never runs — a failure in
        # the backend testsets (defined first) skips the entire compute
        # suite, and the partial results table reads as a mostly-green run
        # (issue #403). Make the abort say what it skipped.
        println()
        printstyled(
            "ERROR: the test run ABORTED at the first failing testset.\n";
            color = :red, bold = true)
        println(
            "Any testset absent from the results table above NEVER RAN.\n",
            "The full list of testsets this run was supposed to execute is:")
        retest(JACCTests; dry = true)
        rethrow()
    end
end
