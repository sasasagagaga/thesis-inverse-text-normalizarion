class DammyLRScheduler:
    """Тупой lr scheduler (можно использовать, когда для модели в Trainer не подали lr scheduler"""
    def __init__(self, *args, **kwargs):
        pass

    def step(self):
        pass
