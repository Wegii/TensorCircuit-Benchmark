import os
import matplotlib.pyplot as plt
from utils import bcolors


if __name__ == "__main__":
    bs = [512] # [1, 8, 16, 32, 64, 128, 256, 512]
    n_qubits = 9
    n_layers = 3
    
    # Select quantum library
    run_tensorcircuit = True
    run_pennylane = False
    
    # Select device
    cpu = True
    gpu = False

    if run_tensorcircuit:
        print(f"{bcolors.OKGREEN}Running simulations with TensorCircuit library{bcolors.ENDC}")
        
        if cpu:
            # Imports for CPU version
            device = "cpu"
            print(f"{bcolors.WARNING}Selecting CPU as default device{bcolors.ENDC}")
            
            os.environ["JAX_PLATFORMS"] = "cpu"
            import jax
            jax.config.update('jax_platform_name', 'cpu')    
            jax.config.update('jax_default_device', jax.devices('cpu')[0])
        
        if gpu:
            # Imports for GPU version
            device = "gpu"
            print(f"{bcolors.WARNING}Selecting GPU as default device{bcolors.ENDC}")

            # Set third GPU as default device
            os.environ["CUDA_VISIBLE_DEVICES"] = "2"
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            
            import tensorflow as tf
            

        # Import benchmark
        from tc_jax import tc_jax_benchmark
        
        ex_time = []
        print(f"{bcolors.OKBLUE}Simulation started{bcolors.ENDC}")
        for batch_size in bs:
            print(f"Executing with batch_size {batch_size}")
            timing, loss, opt, update = tc_jax_benchmark(batch_size = batch_size,
                                                         n_qubits = n_qubits,
                                                         n_layers = n_layers)
            print("Execution statistics:")
            print(timing)
            print(loss)
            print(opt)
            print(update)
            ex_time.append(timing)

            plt.plot(range(0, len(loss[0])), loss[0])
            plt.xlabel('Execution Step')
            plt.ylabel('Duration')
            plt.grid()
            plt.savefig(f"{device}_evaluation_qb{n_qubits}_nl{n_layers}_circuit_execution_{bs}.png")

        print(f"{bcolors.OKBLUE}Simulation finished{bcolors.ENDC}")
        
        plt.plot(bs, ex_time)
        plt.xlabel('Batch Size')
        plt.ylabel('Duration')
        plt.grid()
        plt.savefig(f"{device}_evaluation_qb{n_qubits}_nl{n_layers}_general_{bs}.png") 
    

    # Pytorch + Pennylane
    if run_pennylane:
        # Import benchmark
        from pl_pt import pl_pt_benchmark
        
        print(f"{bcolors.OKGREEN}Running simulations with Pennylane library{bcolors.ENDC}")
        print(f"{bcolors.WARNING}Selecting CPU as default device{bcolors.ENDC}")

        print(f"{bcolors.OKBLUE}Simulation started{bcolors.ENDC}")
        for batch_size in bs:
            print(f"Executing with batch_size {batch_size}")
            pl_pt_benchmark(batch_size = batch_size,
                            n_qubits = n_qubits,
                            n_layers = n_layers)
        print(f"{bcolors.OKBLUE}Simulation finished{bcolors.ENDC}")


        
