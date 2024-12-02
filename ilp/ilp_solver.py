from ortools.init.python import init
from ortools.linear_solver import pywraplp
import json

def main():
    with open("ilp.json", "r") as file:
        ilp_data = json.load(file)
        solver = pywraplp.Solver.CreateSolver("GLOP")
        if not solver:
            print("GLOP solver unavailable.")
            return

        # vars
        var_list = {}
        for var in ilp_data["var_list"]:
            var_list[var] = solver.IntVar(0, 1, var)

        # objective
        obj = solver.Objective()
        for term in ilp_data["objective"]:
            obj.SetCoefficient(var_list[term["indicator"]], term["cost"])
        obj.SetMinimization()

        # constraints
        infinity = solver.infinity()
        for sum in ilp_data["reg_assign"]:
            constraint = solver.Constraint(1, infinity)
            for term in sum:
                constraint.SetCoefficient(var_list[term], 1)

        for term in ilp_data["reg_move"]:
            constraint = solver.Constraint(0, infinity)
            constraint.SetCoefficient(var_list[term["x1"]], 1)
            constraint.SetCoefficient(var_list[term["x2"]], -1)
            constraint.SetCoefficient(var_list[term["c"]], 1)
        
        result_status = solver.Solve()

        print(f"Status: {result_status}")
        if result_status != pywraplp.Solver.OPTIMAL:
            print("The problem does not have an optimal solution!")
            if result_status == pywraplp.Solver.FEASIBLE:
                print("A potentially suboptimal solution was found")
            else:
                print("The solver could not solve the problem.")
                return

        print("Solution:")
        print("Objective value =", obj.Value())
        # print("x =", x_var.solution_value())
        # print("y =", y_var.solution_value())

        print("Advanced usage:")
        print(f"Problem solved in {solver.wall_time():d} milliseconds")
        print(f"Problem solved in {solver.iterations():d} iterations")


if __name__ == "__main__":
    init.CppBridge.init_logging("ilp_solver.py")
    cpp_flags = init.CppFlags()
    cpp_flags.stderrthreshold = True
    cpp_flags.log_prefix = False
    init.CppBridge.set_flags(cpp_flags)
    main()