import rclpy
from rclpy.node import Node

from memory_service_interfaces.srv import UpdateMemory, GetMemory

from memory_service.memory_manager_llm import MemoryAgent
from memory_service.consolidation import (
    serialize_core_memory_for_prompt,
    serialize_core_memory_ids,
    serialize_operation_log_for_response,
)


def to_string_list(values):
    """Coerce an iterable into the list[str] expected by the ROS interfaces.

    ROS refuses anything that is not a plain string, and message contents coming
    from an LLM are not always strings (some providers return a list of content
    blocks).
    """
    if not values:
        return []
    return [value if isinstance(value, str) else str(value) for value in values]


class MemoryServer(Node):
    def __init__(self, agent=None):
        super().__init__('memory_server')

        # The agent can be injected to run the node against a test double.
        self.memory_agent = agent or MemoryAgent()

        # Create the two services
        self.update_service = self.create_service(
            UpdateMemory, 'update_memory', self.update_memory_callback)
        self.get_service = self.create_service(
            GetMemory, 'get_memory', self.get_memory_callback)

    def _fill_core_memory(self, response, state):
        """Publish the active core memories as two aligned arrays.

        memory_list carries the contents, memory_ids the identifiers: same
        filtering and same order, so a client can zip them and refer to a
        specific memory afterwards.
        """
        core_memory = state.get("core_memory", [])
        response.memory_list = to_string_list(serialize_core_memory_for_prompt(core_memory))
        response.memory_ids = to_string_list(serialize_core_memory_ids(core_memory))

    def _fill_operation_log(self, response):
        """Publish what the consolidation did during this call.

        Only the operations of this run: the full log grows for the whole life of
        the node, and get_memory is called often enough that sending the entire
        history every time would cost more with every interaction.
        """
        operations = []
        if hasattr(self.memory_agent, "last_operations"):
            operations = self.memory_agent.last_operations()
        response.operation_log = to_string_list(
            serialize_operation_log_for_response(operations))

    def update_memory_callback(self, request, response):
        self.get_logger().info(
            f"UpdateMemory request: user_input={request.user_input}, "
            f"response={request.explanation}")

        try:
            # Append messages
            self.memory_agent.append_message(request.user_input, 'user')
            self.memory_agent.append_message(request.explanation, 'assistant')

            # Run the agent and get the state dict
            state = self.memory_agent.run_memory_agent(interaction_mode='insert')

            # Populate the response object
            self._fill_core_memory(response, state)
            self._fill_operation_log(response)

            self.get_logger().info(f"Core memory: {response.memory_list}")
            self.get_logger().info(
                f"Operations performed: {len(response.operation_log)}")

            # Return the response object
            return response

        except Exception as e:
            self.get_logger().error(f"UpdateMemory error: {e}")
            response.memory_list = []
            response.memory_ids = []
            response.operation_log = []
            return response

    def get_memory_callback(self, request, response):
        self.get_logger().info("GetMemory request")

        try:
            # Run the agent in retrieve mode, on the message the user just sent:
            # senza, il recupero cercherebbe in archivio partendo dall'ultimo
            # messaggio gia' in memoria invece che dalla domanda corrente.
            state = self.memory_agent.run_memory_agent(
                interaction_mode='retrieve', query=request.user_input)

            # Populate the response object
            self._fill_core_memory(response, state)
            self._fill_operation_log(response)
            response.last_messages = to_string_list(
                [message.content for message in state.get("messages", [])])

            self.get_logger().info(f"Last messages: {response.last_messages}")
            self.get_logger().info(f"Returning memory_list: {response.memory_list}")

            return response

        except Exception as e:
            self.get_logger().error(f"GetMemory error: {e}")
            response.memory_list = []
            response.memory_ids = []
            response.last_messages = []
            response.operation_log = []
            return response


def main(args=None):
    rclpy.init(args=args)
    memory_server = MemoryServer()
    try:
        rclpy.spin(memory_server)
    except KeyboardInterrupt:
        pass
    finally:
        memory_server.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
