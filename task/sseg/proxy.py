import pixelssl

import func, data, model, criterion


def add_parser_arguments(parser):
    pixelssl.proxy_template.add_parser_arguments(parser)

    data.add_parser_arguments(parser)
    model.add_parser_arguments(parser)
    criterion.add_parser_arguments(parser)

    parser.add_argument('--num-classes', type=int, default=21, help='sseg - number of categories in semantic segmentation')
    parser.add_argument('--ignore-index', type=int, default=255, 
                        help='sseg - pixels with the target value are ignored, such as pixels belonging to semantic boundaries')


class SemanticSegmentationProxy(pixelssl.proxy_template.TaskProxy):
    
    NAME = 'sseg'
    TASK_TYPE = pixelssl.CLASSIFICATION

    def __init__(self, args):
        super(SemanticSegmentationProxy, self).__init__(args, func, data, model, criterion)
