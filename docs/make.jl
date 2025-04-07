using FreeEnergyMachine
using Documenter

DocMeta.setdocmeta!(FreeEnergyMachine, :DocTestSetup, :(using FreeEnergyMachine); recursive=true)

makedocs(;
    modules=[FreeEnergyMachine],
    authors="Xiwei Pan",
    sitename="FreeEnergyMachine.jl",
    format=Documenter.HTML(;
        canonical="https://isPANN.github.io/FreeEnergyMachine.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/isPANN/FreeEnergyMachine.jl",
    devbranch="main",
)
