# -*- coding: utf-8 -*-
"""
Диагностика TPU и multiprocessing.
Запусти в Colab: !python debug_tpu.py
"""
import os
import sys
import time
import multiprocessing as mp

print("=" * 60)
print("ДИАГНОСТИКА TPU")
print("=" * 60)

# 1. Проверка TPU
print("\n1. Проверка TPU...")
try:
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    
    device = xm.xla_device()
    print(f"   ✅ TPU доступен: {device}")
    
    # Тест вычислений на TPU
    x = torch.randn(10, 10, device=device)
    y = x @ x.T
    xm.mark_step()
    print(f"   ✅ TPU вычисления работают: {y.shape}")
    TPU_OK = True
except Exception as e:
    print(f"   ❌ TPU ошибка: {e}")
    TPU_OK = False

# 2. Проверка multiprocessing
print("\n2. Проверка multiprocessing...")

def simple_worker(worker_id, queue):
    """Простой воркер."""
    try:
        queue.put(f"Worker {worker_id}: PID={os.getpid()}")
    except Exception as e:
        queue.put(f"Worker {worker_id} ERROR: {e}")

try:
    queue = mp.Queue()
    processes = []
    
    for i in range(3):
        p = mp.Process(target=simple_worker, args=(i, queue))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join(timeout=5)
    
    results = []
    while not queue.empty():
        results.append(queue.get_nowait())
    
    print(f"   ✅ Multiprocessing работает: {results}")
    MP_OK = True
except Exception as e:
    print(f"   ❌ Multiprocessing ошибка: {e}")
    MP_OK = False

# 3. Проверка TPU в дочернем процессе
print("\n3. Проверка TPU в дочернем процессе...")

def tpu_worker(queue):
    """Воркер с TPU."""
    try:
        import torch
        import torch_xla.core.xla_model as xm
        
        device = xm.xla_device()
        x = torch.randn(5, 5, device=device)
        y = x + x
        xm.mark_step()
        
        queue.put(f"TPU worker OK: device={device}, result_shape={y.shape}")
    except Exception as e:
        import traceback
        queue.put(f"TPU worker ERROR: {e}\n{traceback.format_exc()}")

try:
    queue = mp.Queue()
    p = mp.Process(target=tpu_worker, args=(queue,))
    p.start()
    p.join(timeout=30)
    
    if not queue.empty():
        result = queue.get()
        if "ERROR" in result:
            print(f"   ❌ {result}")
            TPU_CHILD_OK = False
        else:
            print(f"   ✅ {result}")
            TPU_CHILD_OK = True
    else:
        print("   ❌ Воркер не ответил (timeout или crash)")
        TPU_CHILD_OK = False
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
    TPU_CHILD_OK = False

# 4. Проверка нашей модели
print("\n4. Проверка загрузки модели...")
try:
    sys.path.insert(0, '.')
    from rl_chess.RL_network import ChessNetwork
    
    model = ChessNetwork()
    print(f"   ✅ Модель создана: {sum(p.numel() for p in model.parameters())} параметров")
    
    if TPU_OK:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        model = model.to(device)
        print(f"   ✅ Модель на TPU")
except Exception as e:
    print(f"   ❌ Ошибка модели: {e}")

# 5. Проверка InferenceServer
print("\n5. Проверка InferenceServer...")

def inference_server_test(input_queue, output_queue):
    """Тест InferenceServer."""
    try:
        import torch
        import torch_xla.core.xla_model as xm
        from rl_chess.RL_network import ChessNetwork
        
        device = xm.xla_device()
        model = ChessNetwork().to(device)
        model.eval()
        
        output_queue.put("SERVER_READY")
        
        # Ждём запрос
        req = input_queue.get(timeout=10)
        
        # Делаем инференс
        with torch.no_grad():
            dummy_input = torch.randn(1, 102, 8, 8, device=device)
            policy, value = model(dummy_input)
            xm.mark_step()
        
        output_queue.put(f"INFERENCE_OK: policy={policy.shape}, value={value.shape}")
        
    except Exception as e:
        import traceback
        output_queue.put(f"SERVER_ERROR: {e}\n{traceback.format_exc()}")

try:
    input_q = mp.Queue()
    output_q = mp.Queue()
    
    server = mp.Process(target=inference_server_test, args=(input_q, output_q))
    server.start()
    
    # Ждём готовности
    time.sleep(3)
    
    if not output_q.empty():
        msg = output_q.get()
        if msg == "SERVER_READY":
            print("   ✅ Сервер запустился")
            
            # Отправляем тестовый запрос
            input_q.put("TEST_REQUEST")
            
            time.sleep(5)
            if not output_q.empty():
                result = output_q.get()
                if "ERROR" in result:
                    print(f"   ❌ {result}")
                else:
                    print(f"   ✅ {result}")
            else:
                print("   ❌ Сервер не ответил на запрос")
        else:
            print(f"   ❌ {msg}")
    else:
        print("   ❌ Сервер не запустился")
    
    server.terminate()
    server.join(timeout=5)
    
except Exception as e:
    import traceback
    print(f"   ❌ Ошибка: {e}\n{traceback.format_exc()}")

# Итоги
print("\n" + "=" * 60)
print("ИТОГИ:")
print("=" * 60)
print(f"TPU в главном процессе:  {'✅' if TPU_OK else '❌'}")
print(f"Multiprocessing:         {'✅' if MP_OK else '❌'}")
print(f"TPU в дочернем процессе: {'✅' if TPU_CHILD_OK else '❌'}")
print("=" * 60)

if not TPU_CHILD_OK:
    print("\n⚠️ ПРОБЛЕМА: TPU не работает в дочерних процессах!")
    print("   Это известное ограничение TPU + multiprocessing.")
    print("   Решение: использовать однопроцессный режим (main_train.py)")
