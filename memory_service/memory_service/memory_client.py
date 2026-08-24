import json

import rclpy
from rclpy.node import Node

from memory_service_interfaces.srv import UpdateMemory, GetMemory


class MemoryClient(Node):
    def __init__(self):
        super().__init__('memory_client')
        self.update_client = self.create_client(UpdateMemory, 'update_memory')
        self.get_client = self.create_client(GetMemory, 'get_memory')

        while not self.update_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for update_memory service...')
        while not self.get_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for get_memory service...')

    def send_update_request(self, user_input, explanation, queries=None, results=''):
        req = UpdateMemory.Request()
        req.user_input = str(user_input)
        req.queries = [str(q) for q in (queries or [])]
        req.results = str(results)
        req.explanation = str(explanation)
        future = self.update_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def send_get_request(self):
        req = GetMemory.Request()
        future = self.get_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


def print_memory(title, memory_list, memory_ids):
    """Core memories with their identifier, since the two arrays are aligned."""
    print(f'{title} ({len(memory_list)} memorie attive):')
    if not memory_list:
        print('  (core memory vuota)')
    for content, item_id in zip(memory_list, memory_ids):
        print(f'  - [{item_id}] {content}')


def print_operations(operation_log):
    """What the consolidation did during the call, one JSON entry per line."""
    print(f'Operazioni di consolidamento ({len(operation_log)}):')
    if not operation_log:
        print('  (nessuna operazione)')
    for entry in operation_log:
        operation = json.loads(entry)
        related = operation.get('related_item_id') or '-'
        print(f"  - {operation['op_type']:<10} | item: {operation['item_id']}"
              f" | related: {related} | {operation.get('content')}")


def main(args=None):
    rclpy.init(args=args)
    client = MemoryClient()

    # Example usage
    update_response = client.send_update_request(
        'ciao voglio mangiare la nutella',
        'puoi mangiare la nutella',
        queries=['q1', 'q2'],
        results='result1, result2',
    )
    print_memory('Dopo update', update_response.memory_list, update_response.memory_ids)
    print_operations(update_response.operation_log)

    get_response = client.send_get_request()
    print()
    print_memory('Memoria corrente', get_response.memory_list, get_response.memory_ids)
    print('Ultimi messaggi:', list(get_response.last_messages))
    print_operations(get_response.operation_log)

    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
