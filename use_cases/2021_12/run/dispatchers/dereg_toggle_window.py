"""
Custom Dispatcher for 2021 LWRS FPOG Milestone studying the effects of
different storage technologies in the NYISO market.
"""
from functools import partial

import os
import sys
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
try:
  import _utils as hutils
except (ModuleNotFoundError, ImportError):
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
  import _utils as hutils

# Global Constants
TIME_WINDOW = 24
PICARD_LIMIT = 10
ECONV_RATE = 0.33
HEAT_TECHS = ['Hitec_XL_storage', 'Dowtherm_A_storage']
E_TECHS = ['ETES_storage', 'Li_ion_storage', 'H2_storage',]
OTHER_COMPS = ['Additional_NPP', 'NPP_unused', 'turbine']

# This is what the NYISO stack looks like:
#                                            capacity_gw     marginal_cost
# NY    Reference    Storage    Solar        14.289311       0.000000
#                               Wind         10.435570       0.000000
#                               Hydro         4.277016       0.000000
#                               Nuclear       1.143500      12.689066
#                               NGCC          4.533380      25.109150
#                               NGGT         17.224408      32.703513
#                               Overflow     35.000000     169.679554
#                    No_Storage Solar        14.894274       0.000000
#                               Wind         10.435570       0.000000
#                               Hydro         4.277016       0.000000
#                               Nuclear       1.143500      12.689066
#                               NGCC          4.533380      25.109150
#                               NGGT         14.524706      32.703513
#                               Overflow     35.000000     169.679554
#       CES          Storage    Solar         9.774715       0.000000
#                               Wind         11.881870       0.000000
#                               Hydro         4.277016       0.000000
#                               Nuclear       5.242805      12.689066
#                               NGCC          2.101907      25.109150
#                               NGGT          4.638107      32.703513
#                               Overflow     35.000000     169.679554
#                    No_Storage Solar         9.774715       0.000000
#                               Wind         11.881869       0.000000
#                               Hydro         4.277016       0.000000
#                               Nuclear       5.242805      12.689066
#                               NGCC          2.101907      25.109150
#                               NGGT          2.101907      32.703513
#                               Overflow     35.000000     169.679554

# 'coeffs' taken from exponential fit formulas computed in
# ../../scripts/cap_cost_proc.py script
CASE_DICT = {('Reference', 'No_Storage'): {'coeffs': (12.791, 0.025),
                                           'windcap': 10.435570,
                                           'solarcap': 14.894274,
                                           'nucap': 1.143500},

              ('Reference', 'Storage'):    {'coeffs': (12.881, 0.026),
                                            'windcap': 10.435570,
                                            'solarcap': 14.289311,
                                            'nucap': 1.143500},

              ('CES', 'No_Storage'):       {'coeffs': (12.403, 0.032),
                                            'windcap': 11.881869,
                                            'solarcap': 9.774715,
                                            'nucap': 5.242805},

              ('CES', 'Storage'):          {'coeffs': (12.791, 0.034),
                                            'windcap': 9.774715,
                                            'solarcap': 11.881870,
                                            'nucap': 5.242805}}


def dispatch(info, activity_matrix):
  """
  Dispatches the components based on user-defined algorithms.
  The expected return object is a dict of components, mapped to a dict of
    resources that component uses, mapped to the amount consumed/produced
    as a numpy array.
  Note:
    - Negative values mean the component consumes that resource
    - Positive values mean the component produces that resource
    - The activity doesn't necessarily have to be as long as "time", but it usually should be
  @ In, info, dict, information about the state of the system
  @ Out, activity, dict, activity of components as described above
  """
  time = getattr(activity_matrix, "_times")
  heron = info['HERON']
  case = info['HERON']['Case']
  components = getattr(activity_matrix, "_components")
  sources = heron['Sources']
  labels = heron['Case'].get_labels()
  label_id = (labels['price_struct'], labels['strategy'])
  resources = sorted(list(hutils.get_all_resources(components))) # list of all active resources

  if 'time_window_length' in labels.keys():
    time_window = labels['time_window_length']
  else:
    time_window = TIME_WINDOW

  # rolling window
  start_index = 0
  final_index = len(time)
  # TODO window overlap!  ( )[ ] -> (   [  )   ]
  while start_index < final_index:
    end_index = start_index + int(time_window)
    if end_index > final_index:
      end_index = final_index
    if end_index - start_index == 1:
      # TODO custom error raise for catching in DispatchManager?
      raise IOError("A rolling window of length 1 was requested, but this causes crashes in pyomo. " +
                    "Change the length of the rolling window to avoid length 1 histories.")
    specific_time = time[start_index:end_index]
    print('DEBUGG starting window {} to {}'.format(start_index, end_index))

    # set initial storage levels
    initial_levels = {}
    for comp in components:
      if comp.get_interaction().is_type('Storage'):
        if start_index == 0:
          initial_levels[comp] = comp.get_interaction().get_initial_level(info)
        else:
          init_lvl = subdisp[comp.name]['level'][comp.get_interaction().get_resource()][-1]
          initial_levels[comp] = init_lvl if init_lvl>=0 else 0
    # allow for converging solution iteratively
    converged = False
    conv_counter = 0
    previous = None
    while not converged:
      conv_counter += 1
      if conv_counter > PICARD_LIMIT:
        break
      # dispatch
      subdisp = dispatch_window(specific_time, start_index,
                                      case, components, sources, resources,
                                      initial_levels, info)
      # do we need a convergence criteria? Check now.
      if needs_convergence(components):
        print(f'DEBUGG iteratively solving window, iteration {conv_counter}/{PICARD_LIMIT} ...')
        converged = check_converged(subdisp, previous, components)
        previous = subdisp
      else:
        converged = True
    # store result in corresponding part of dispatch
    for comp in components:
      for tag in comp.get_tracking_vars():
        for res, values in subdisp[comp.name][tag].items():
          activity_matrix.set_activity_vector(comp, res, values, tracker=tag, start_idx=start_index, end_idx=end_index)
    start_index = end_index
  return activity_matrix

def dispatch_window(specific_time, start_index,
                    case, components, sources, resources,
                    initial_levels, info):
    """
      Dispatches one part of a rolling window.
      @ In, time, np.array, value of time to evaluate
      @ In, time_offset, int, offset of the time index in the greater history
      @ In, case, HERON Case, Case that this dispatch is part of
      @ In, components, list, HERON components available to the dispatch
      @ In, sources, list, HERON source (placeholders) for signals
      @ In, resources, list, sorted list of all resources in problem
      @ In, initial_storage, dict, initial storage levels if any
      @ In, meta, dict, additional variables passed through
      @ Out, result, dict, results of window dispatch
    """
    result = {}
    heron = info['HERON']
    labels = heron['Case'].get_labels()
    label_id = (labels['price_struct'], labels['strategy'])
    components = dict((c.name, c) for c in heron['Components'])

    # Grab wind/solar capacity and wind/solar generation
    wind_cap = CASE_DICT[(labels['price_struct'], labels['strategy'])]['windcap']
    solar_cap = CASE_DICT[(labels['price_struct'], labels['strategy'])]['solarcap']
    wind_pct = heron['RAVEN_vars']['WIND']
    solar_pct = heron['RAVEN_vars']['SOLAR']
    total_load = heron['RAVEN_vars']['TOTALLOAD']
    wind = wind_pct * wind_cap
    solar = solar_pct * solar_cap
    # This is the net load after removing VREs
    load = total_load - wind - solar
    load[load < 0] = 0 # Curtail VRE Sources

    # Since ARMA resolution is by hour we hold this constant
    time_dt = 1
    T = len(specific_time)
    time_slice = slice(start_index, start_index+T)
    storage_dat = dict(
        (name, {
            'initial_level': float(components[name].get_interaction().get_initial_level(info)),
            'capacity': components[name].get_capacity(info)[0][components[name].get_interaction().get_resource()],
            'component': components[name],
            'RTE': float(components[name].get_sqrt_RTE())
        }) for name in HEAT_TECHS + E_TECHS)

    ###
    # PYOMO MODEL
    ###
    m = pyo.ConcreteModel()
    Ts = np.arange(0, T, dtype=int)
    m.T = pyo.Set(initialize=Ts)
    m.Load = load[time_slice]
    m.StorageDat = storage_dat
    m.Labels = label_id

    # Components: Nuclear, Turbines, TES, Grid
    add_npp = components['Additional_NPP'].get_capacity(info)[0]['heat']
    # We must divide by ECONV_RATE here because EPRI provided nucap in electricity
    # but since additional npp is in heat, we must convert EPRI's capacity data.
    total_npp = (CASE_DICT[label_id]['nucap'] / ECONV_RATE) + add_npp
    m.NuCap = total_npp
    # Our clearing price function is trained on the baseline
    m.BaselineNuCap = (CASE_DICT[label_id]['nucap'] / ECONV_RATE)

    m.Turbine = pyo.Var(m.T,
                        within=pyo.NonNegativeReals,
                        bounds=(0, max(load) * 1.1), # Upper Bound is > leftover load
                        initialize=0)

    m.NPP_unused = pyo.Var(m.T, within=pyo.NonPositiveReals, initialize=0)

    # Create 3 pyomo vars for each TES
    for tes in HEAT_TECHS + E_TECHS:
        cap = storage_dat[tes]['capacity']
        level = pyo.Var(m.T,
                        within=pyo.NonNegativeReals,
                        bounds=(0, cap),
                        initialize=initial_levels[components[tes]])
        charge = pyo.Var(m.T,
                         within=pyo.NonPositiveReals,
                         bounds=(-cap/time_dt, 0),
                         initialize=0)
        discharge = pyo.Var(m.T,
                            within=pyo.NonNegativeReals,
                            bounds=(0, cap/time_dt),
                            initialize=0)
        setattr(m, f'{tes}_level', level)
        setattr(m, f'{tes}_charge', charge)
        setattr(m, f'{tes}_discharge', discharge)

    m.Other = pyo.Var(m.T, within=pyo.NonNegativeReals)

    # Conservation Contraint Functions
    m.EConserve = pyo.Constraint(m.T, rule=_conserve_elec)
    m.HConserve = pyo.Constraint(m.T, rule=_conserve_heat)

    # Clean H2 Constraints (Used in Tax Credit Case)
    # m.CleanH = pyo.Constraint(m.T, rule=_clean_h2)

    # Transfer Functions
    for tech in HEAT_TECHS + E_TECHS:
        func = partial(_storage_mgmt_level, tech)
        setattr(m, f'{tech}_mgmt_level', pyo.Constraint(m.T, rule=func))

    # Objective Function
    m.OBJ = pyo.Objective(sense=pyo.maximize, rule=_objective)

    # m.pprint()
    soln = pyo.SolverFactory('ipopt').solve(m)
    if soln.solver.status == SolverStatus.ok and soln.solver.termination_condition == TerminationCondition.optimal:
        print('Successful optimization solve.')
        # debug_pyomo_soln(m)
    else:
        print('Storage optimization FAILURE!')
        print(' ... status:', soln.solver.status)
        print(' ... termination:', soln.solver.termination_condition)
        # m.pprint()
        raise RuntimeError('Failed solve!')

    # return dict of numpy arrays
    result = retrieve_solution(m, add_npp)
    return result


def retrieve_solution(m, add_npp):
  """
    Extracts solution from Pyomo optimization
    @ In, m, pyo.ConcreteModel, associated (solved) model
    @ Out, result, dict, {comp: {activity: {resource: [production]}} e.g. generator[production][steam]
  """
  result = {} # {component: {activity_tag: {resource: production}}}

  # store activity
  # NOTE for SecM and E_dump we directly include prices here rather than recalculate them in TEAL,
  #      since we have the load data now and won't have it (as accessible) then.
  for htech in HEAT_TECHS:
    result[htech] = {}
    for trackr in ['level', 'charge', 'discharge']:
      tag_values = {}
      tag_values['heat'] = np.array(list(getattr(m, f'{htech}_{trackr}')[t].value for t in m.T))
      result[htech][trackr] = tag_values

  # 'grid_sales' is electricity sold to grid coming from NPP, Electrical Storage, Thermal Storage.
  # However, the turbine output already incorporates elec from the NPP and Thermal Storage, so we just
  # need to gather the electricity from Electrical Storage.
  grid_sales = np.zeros(len(m.T))
  for etech in E_TECHS:
    result[etech] = {}
    for trackr in ['level', 'charge', 'discharge']:
      tag_values = {}
      act = np.array(list(getattr(m, f'{etech}_{trackr}')[t].value for t in m.T))
      tag_values['electricity'] = act
      result[etech][trackr] = tag_values
      # 'charge' is in both 'charge' and 'discharge' so this is a lil' dirty.
      if 'charge' in trackr:
        grid_sales += act

  for comp in OTHER_COMPS:
    result[comp] = {}
    result[comp]['production'] = {}
    if comp == 'Additional_NPP':
      npp_values = {}
      npp_values['heat'] = np.ones(len(m.T)) * add_npp
      result[comp]['production'] = npp_values
    elif comp == 'NPP_unused':
      npp_unused_heat = {}
      npp_unused_heat['heat'] = np.array(list(m.NPP_unused[t].value for t in m.T))
      result[comp]['production'] = npp_unused_heat
    elif comp == 'turbine':
      turbine_elec_out = {}
      turbine_elec_in = {}
      turb_val = np.array(list(m.Turbine[t].value for t in m.T))
      turbine_elec_out['electricity'] = turb_val
      turbine_elec_in['heat'] = -turb_val / ECONV_RATE
      result[comp]['production'] = turbine_elec_out
      result[comp]['production'] = turbine_elec_in

  # We flip the sign because grid_sales should be all positive and grid is consuming.
  grid_sales += turb_val # (npp_unused_heat / ECONV_RATE)
  result['grid'] = {}
  result['grid']['production'] = {}
  result['grid']['production']['electricity'] = -grid_sales

  return result

def _objective(m):
    """
    TODO
    """
    a, b = CASE_DICT[m.Labels]['coeffs']
    total = 0
    for t in m.T:
        turbine_prod = m.Turbine[t]
        expected_npp = (m.BaselineNuCap * ECONV_RATE)

        # Gather the production of the electric batteries
        li_ion_prod = m.Li_ion_storage_charge[t] + m.Li_ion_storage_discharge[t]
        h2_prod = m.H2_storage_charge[t] + m.H2_storage_discharge[t]
        etes_prod = m.ETES_storage_charge[t] + m.ETES_storage_discharge[t]

        # X is the load that needs to be met by non-VRE, non-storage sources,
        # assuming baseline nuc availability.
        # So, we need to subtract off all the unexpected storage and extra nuclear.
        x = m.Load[t] - (li_ion_prod + h2_prod + etes_prod) - (turbine_prod - expected_npp)

        # x can never be less than zero - Proof by Contradiction
        # l - (p - n) < 0 st p<=l, l>=0, p>=0, n>=0
        # l < p - n
        # l + n < p ~ This can't be true

        # This curve represents the clearing price stack fitted by an exponential function.
        clearing_price = a * pyo.exp(b * x) * 1e3

        # Economic Drivers
        # 1. Profit of E-Storage Selling E
        # 2. Profit of Nuclear for Selling E
        # 3. Offsetting VRE to allow Nuclear to operate fully
        total += clearing_price * (li_ion_prod + h2_prod + etes_prod + turbine_prod)
        total += 5610 * m.NPP_unused[t] # 5610 $/GWth
    return total


def _storage_mgmt_level(name, m, t):
    """
    TODO
    """
    level = getattr(m, f'{name}_level')
    charge = getattr(m, f'{name}_charge')
    discharge = getattr(m, f'{name}_discharge')
    if t > 0:
        prev = level[t-1]
    else:
        prev = m.StorageDat[name]['initial_level']
    sqrt_rte = m.StorageDat[name]['RTE']
    prod = (-sqrt_rte * charge[t]) - (discharge[t] / sqrt_rte)
    # we multiply by dt here but it's 1 always
    return level[t] == prev + prod * 1


# def _clean_h2(m, t):
#     """
#     TODO
#     """
#     charge = getattr(m, 'H2_storage_charge')[t]
#     return -charge <= m.Turbine[t]

def _conserve_elec(m, t):
    """
    TODO
    """
    sources = m.Turbine[t] + m.Other[t]
    sinks = -m.Load[t]
    battery = 0
    for es in E_TECHS:
        charge = getattr(m, f'{es}_charge')[t]
        discharge = getattr(m, f'{es}_discharge')[t]
        battery += (charge + discharge)
    return sources + battery + sinks == 0


def _conserve_heat(m, t):
    """
    TODO
    """
    sources = m.NuCap
    sinks = -(m.Turbine[t] / ECONV_RATE) + m.NPP_unused[t]
    battery = 0
    for tes in HEAT_TECHS:
        charge = getattr(m, f'{tes}_charge')[t]
        discharge = getattr(m, f'{tes}_discharge')[t]
        battery += (charge + discharge)
    return sources + battery + sinks == 0

def needs_convergence(components):
  """
    Determines whether the current setup needs convergence to solve.
    @ In, components, list, HERON component list
    @ Out, needs_convergence, bool, True if iteration is needed
  """
  for comp in components:
    intr = comp.get_interaction()
    # storages with a prescribed strategy MAY need iteration
    if intr.is_governed():
      return True
  # if we get here, no iteration is needed
  return False

def check_converged(new, old, components):
  """
    Checks convergence of consecutive dispatch solves
    @ In, new, dict, results of dispatch # TODO should this be the model rather than dict?
    @ In, old, dict, results of previous dispatch
    @ In, components, list, HERON component list
    @ Out, converged, bool, True if convergence is met
  """
  tol = 1e-4 # TODO user option
  if old is None:
    return False
  converged = True
  for comp in components:
    intr = comp.get_interaction()
    name = comp.name
    tracker = comp.get_tracking_vars()[0]
    if intr.is_governed(): # by "is_governed" we mean "isn't optimized in pyomo"
      # check activity L2 norm as a differ
      # TODO this may be specific to storage right now
      res = intr.get_resource()
      scale = np.max(old[name][tracker][res])
      diff = np.linalg.norm(new[name][tracker][res] - old[name][tracker][res]) / (scale if scale != 0 else 1)
      if diff > tol:
        converged = False
  return converged

def debug_pyomo_print(m):
    print('/' + '='*80)
    print('DEBUGG model pieces:')
    print('  -> objective:')
    # print('     ', m.objective.pprint())
    print('  -> variables:')
    for var in m.component_objects(pyo.Var):
        # print('     ', var.pprint())
        print('  -> constraints:')
    for constr in m.component_objects(pyo.Constraint):
        # print('     ', constr.pprint())
        print('\\' + '='*80)
        print('')


def debug_pyomo_soln(m):
    print('DEBUGG solution:')
    print('  -> objective:', m.OBJ())
    # print('t: ' + ', '.join(f'{var.name:9s}' for var in m.component_objects(pyo.Var)))
    for t in m.T:
        msg = f'{t}: '
        for var in m.component_objects(pyo.Var):
            msg += f'{var[t].value:1.3e}, '
            # print(msg)


def debug_setup_print(activity, m, stack, prices, comp_order, load,
                      htse_cap_h2, npp_cap, store_cap, store_initial, market_cap,
                      price_h2_kg, price_h2_GW, NPP_bid, H2_opp, bid_adjust):
    print('DEBUGG capacities:')
    print(f'  -> npp (GW): {npp_cap:1.2e}')
    print(f'  -> store (kgH2): {store_cap:1.2e}')
    print(f'  -> store initial (kgH2): {store_initial:1.2e}')
    print(f'  -> market (kgH2/s): {market_cap:1.2e}')
    print('DEBUGG conversion/costs:')
    print(f'  -> HTSE e/H2 (GW/kg/s): {e_per_h2:1.2e}')
    print(f'  -> H2 price ($/kg/s): {price_h2_kg:1.2e}')
    print(f'  -> H2_opp ($/GW): {H2_opp:1.2e}')
    print(f'  -> NPP bid adjust ($/GW): {float(bid_adjust):1.2e}')
    print(f'  -> NPP TOTAL bid ($/GW): {float(NPP_bid):1.2e}')
    print('DEBUGG activity:')
    print('  t: load(GW), NPP->grid(GW)') #, htse(H2/s), store(H2)')
    template = '  {t}: {load:1.2e}, {grid:1.2e}' #, {htse:1.2e}, {store:1.2e}'
    for t in range(len(load)):
        print(template.format(t=t, load=load[t],
                              # npp=activity['NPP']['electricity'][t],
                              grid=activity['grid']['electricity'][t],
                              # htse=activity['HTSE']['H2'][t],
                              # store=activity['H2_storage']['H2'][t],
                              ))
        print('DEBUGG stack:')
        print('  i: stack, price, comp')
        for i in range(len(stack)):
            print(f'  {i}: {stack[i]:1.2e}, {prices[i]:1.2e}, {comp_order[i]}')
